import os
import json
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# openaiのインポート
import openai
from langchain_community.vectorstores import Chroma  # ★FAISSからChromaに変更
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

# 1. Pydanticモデル定義 (変更なし)
class EstimateItem(BaseModel):
    name: str = Field(..., description="項目名")
    description: str = Field(..., description="項目の説明")
    unitPrice: int = Field(..., description="単価")
    quantity: int = Field(..., description="数量")
    amount: int = Field(..., description="金額")

class EstimateProposal(BaseModel):
    title: str = Field(..., description="提案タイトル")
    concept: str = Field(..., description="提案コンセプト")
    items: List[EstimateItem] = Field(..., description="見積もり項目リスト")
    subtotal: int = Field(..., description="小計")
    tax: int = Field(..., description="消費税")
    total: int = Field(..., description="合計金額")
    deliveryPeriod: str = Field(..., description="納期")
    notes: str = Field(..., description="備考")

class EstimateResponse(BaseModel):
    proposals: List[EstimateProposal] = Field(..., description="見積もり提案リスト")

class AskRequest(BaseModel):
    question: str
    wanted_plans: Optional[int] = Field(default=2, ge=1, le=5)

# 2. グローバル変数
vectorstore: Optional[Chroma] = None  # ★FAISSからChromaに変更
df_vectordb: Optional[pd.DataFrame] = None
is_ai_ready = False

# 3. FastAPIアプリケーションのインスタンス化
app = FastAPI(title="AI Ito-san API", version="5.0 Final Stable")

# 4. GASロジックを再現するPython関数群
def correct_and_summarize(text: str) -> str:
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'system', 'content': 'あなたは優秀な日本語校正者です。受け取った文章の誤字脱字を修正し、意味を変えずに200字以内で要約してください。'}, 
            {'role': 'user', 'content': text}
        ],
        max_tokens=300, 
        temperature=0.3
    )
    return response.choices[0].message.content.strip()

def perform_similarity_search(original_text: str) -> List[str]:
    # Chromaは同じインターフェースでsimilarity_search_with_scoreが使える
    docs_and_scores = vectorstore.similarity_search_with_score(original_text, k=20)
    similarities = []
    for doc, score in docs_and_scores:
        df_index = doc.metadata['df_index']
        db_row = df_vectordb.loc[df_index]
        similarities.append({
            "id": db_row.get('ID'), 
            "similarity": 1 - score,  # Chromaのスコアも距離なので1から引く
            "category": db_row.get('カテゴリ', 'Knowledge')
        })
    
    similarities.sort(key=lambda x: x['similarity'], reverse=True)
    
    config = {'Estimate': 5, 'Catalog': 5, 'Knowledge': 5}
    selected_ids = []
    for cat, k in config.items():
        cat_items = [s for s in similarities if s['category'] == cat]
        selected_ids.extend([item['id'] for item in cat_items[:k]])
    
    return list(set(selected_ids))

def build_gpt_prompt(corrected_text: str, matched_ids: List[str], wanted_plans: int) -> str:
    reference_texts = {'Estimate': [], 'Catalog': [], 'Knowledge': []}
    
    if matched_ids:
        matched_rows = df_vectordb[df_vectordb['ID'].isin(matched_ids)]
        for _, row in matched_rows.iterrows():
            cat = row.get('カテゴリ', 'Knowledge')
            if cat in reference_texts: 
                reference_texts[cat].append(f"【{row['ID']}】{row['本文']}")
    
    prompt = f"""あなたは顧客の期待を超える見積もりを作る、トップ1%の提案者です。

■依頼内容
{corrected_text}

■参考資料
"""
    
    if reference_texts['Estimate']: 
        prompt += "【過去の見積もり】\n" + "\n".join(reference_texts['Estimate']) + "\n\n"
    if reference_texts['Catalog']: 
        prompt += "【商品カタログ情報】\n" + "\n".join(reference_texts['Catalog']) + "\n\n"
    if reference_texts['Knowledge']: 
        prompt += "【関連ナレッジ】\n" + "\n".join(reference_texts['Knowledge']) + "\n\n"
    
    prompt += f"""
必ず{wanted_plans}案で提案し、以下のJSON形式で回答してください。JSONのみを出力し、他の説明は不要です:

{{
  "proposals": [
    {{
      "title": "提案タイトル",
      "concept": "提案コンセプト（顧客の課題をどう解決するか）", 
      "items": [
        {{
          "name": "項目名",
          "description": "項目の詳細説明",
          "unitPrice": 100000,
          "quantity": 1,
          "amount": 100000
        }}
      ],
      "subtotal": 100000,
      "tax": 10000,
      "total": 110000,
      "deliveryPeriod": "2週間",
      "notes": "特記事項"
    }}
  ]
}}

重要：
- 全ての数値フィールド（unitPrice, quantity, amount, subtotal, tax, total）は数値型で出力
- amount = unitPrice × quantity
- subtotal = 全itemsのamountの合計
- tax = subtotalの10%
- total = subtotal + tax
- {wanted_plans}個の提案を必ず含める
"""
    return prompt

def call_gpt_for_proposal(prompt: str) -> dict:
    response = openai.ChatCompletion.create(
        model='gpt-4',
        messages=[
            {'role': 'system', 'content': 'あなたは見積もり作成の専門家です。指示されたJSONフォーマットで、会話や余計なテキストを一切含めず、JSONオブジェクトのみを回答してください。'}, 
            {'role': 'user', 'content': prompt}
        ],
        max_tokens=4095,
        temperature=0.5
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"proposals": []}

# 5. FastAPIイベントハンドラとエンドポイント
@app.on_event("startup")
def startup_event():
    global vectorstore, df_vectordb, is_ai_ready
    try:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key: 
            raise ValueError("OPENAI_API_KEY is not set.")
        
        # CSVファイルの読み込み
        file_path = "伊藤さん2 - VectorDB (1).csv"
        df_vectordb = pd.read_csv(file_path, engine='python', keep_default_na=False)
        valid_rows = df_vectordb[df_vectordb['本文'].astype(str).str.strip() != ''].copy()
        valid_rows['df_index'] = valid_rows.index
        
        # ドキュメントの作成
        documents = [
            Document(
                page_content=row['本文'], 
                metadata={'df_index': idx}
            ) for idx, row in valid_rows.iterrows()
        ]
        
        if not documents: 
            raise ValueError("No valid documents found in CSV.")
        
        # Embeddingsモデルの初期化
        embeddings_model = OpenAIEmbeddings(openai_api_key=openai.api_key)
        
        # ★Chromaでベクトルストアを作成（FAISSと使い方は同じ）
        vectorstore = Chroma.from_documents(
            documents=documents, 
            embedding=embeddings_model,
            persist_directory="./chroma_db"  # Chromaは永続化も簡単
        )
        
        is_ai_ready = True
        print("✅ AIシステムが正常に起動しました。(ChromaDB版)")

    except Exception as e:
        is_ai_ready = False
        import traceback
        print(f"❌ 初期化エラー: {e}")
        traceback.print_exc()

# CORS設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"], 
    allow_headers=["*"]
)

@app.post("/ask", response_model=EstimateResponse)
async def ask_ai_endpoint(request: AskRequest):
    if not is_ai_ready: 
        raise HTTPException(status_code=503, detail="AI is not ready.")
    
    try:
        # 1. テキストの要約
        corrected_text = correct_and_summarize(request.question)
        
        # 2. 類似検索
        matched_ids = perform_similarity_search(request.question)
        
        # デバッグ情報
        print(f"Corrected text: {corrected_text}")
        print(f"Matched IDs: {matched_ids}")
        
        # 3. プロンプト作成
        final_prompt = build_gpt_prompt(
            corrected_text, 
            matched_ids, 
            wanted_plans=request.wanted_plans
        )
        print(f"Final prompt:\n{final_prompt}")
        
        # 4. GPT呼び出し
        proposal_json = call_gpt_for_proposal(final_prompt)
        print(f"GPT response: {proposal_json}")
        
        # 5. エラーハンドリング
        if 'proposals' not in proposal_json or not proposal_json['proposals']:
            proposal_json = {
                "proposals": [{
                    "title": "標準見積もり",
                    "concept": "お客様のご要望に基づいた基本的な提案です",
                    "items": [{
                        "name": "基本作業",
                        "description": "詳細は打ち合わせの上決定",
                        "unitPrice": 100000,
                        "quantity": 1,
                        "amount": 100000
                    }],
                    "subtotal": 100000,
                    "tax": 10000,
                    "total": 110000,
                    "deliveryPeriod": "要相談",
                    "notes": "詳細はお打ち合わせの上決定させていただきます"
                }]
            }
            
        return proposal_json
        
    except Exception as e:
        import traceback
        print(f"回答生成エラー: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def read_root():
    return {"status": "Ready" if is_ai_ready else "Error"}
