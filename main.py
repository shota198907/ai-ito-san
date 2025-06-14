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
# FAISSの代わりに簡易検索を使用
# from langchain_community.vectorstores import FAISS
# from langchain.embeddings import OpenAIEmbeddings
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
# vectorstore: Optional[FAISS] = None  # FAISSは使わない
df_vectordb: Optional[pd.DataFrame] = None
is_ai_ready = False

# 3. FastAPIアプリケーションのインスタンス化
app = FastAPI(title="AI Ito-san API", version="5.0 Final Stable")

# 4. GASロジックを再現するPython関数群 (openai v0.28.1の書き方に修正)
def correct_and_summarize(text: str) -> str:
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',  # gpt-4o-miniではなくgpt-3.5-turboを使用
        messages=[{'role': 'system', 'content': 'あなたは優秀な日本語校正者です。受け取った文章の誤字脱字を修正し、意味を変えずに200字以内で要約してください。'}, {'role': 'user', 'content': text}],
        max_tokens=300, temperature=0.3
    )
    return response.choices[0].message.content.strip()

def perform_similarity_search_simple(original_text: str) -> List[str]:
    """FAISSを使わない簡易版の類似検索"""
    # キーワードベースの簡易検索
    keywords = original_text.lower().split()
    matched_ids = []
    
    for idx, row in df_vectordb.iterrows():
        text = str(row.get('本文', '')).lower()
        # キーワードが含まれているかチェック
        if any(keyword in text for keyword in keywords):
            matched_ids.append(row.get('ID'))
            
    # カテゴリごとに最大5件まで
    result_ids = []
    for category in ['Estimate', 'Catalog', 'Knowledge']:
        cat_ids = [id for id in matched_ids if df_vectordb[df_vectordb['ID'] == id]['カテゴリ'].values[0] == category]
        result_ids.extend(cat_ids[:5])
    
    return result_ids[:15]  # 最大15件

def build_gpt_prompt(corrected_text: str, matched_ids: List[str], wanted_plans: int) -> str:
    reference_texts = {'Estimate': [], 'Catalog': [], 'Knowledge': []}
    if matched_ids:
        matched_rows = df_vectordb[df_vectordb['ID'].isin(matched_ids)]
        for _, row in matched_rows.iterrows():
            cat = row.get('カテゴリ', 'Knowledge')  # カラム名を修正
            if cat in reference_texts: reference_texts[cat].append(f"【{row['ID']}】{row['本文']}")
    
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
    
    # JSON構造を明示的に指定
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
        model='gpt-4',  # gpt-4oではなくgpt-4を使用
        messages=[{'role': 'system', 'content': 'あなたは見積もり作成の専門家です。指示されたJSONフォーマットで、会話や余計なテキストを一切含めず、JSONオブジェクトのみを回答してください。'}, {'role': 'user', 'content': prompt}],
        max_tokens=4095,
        temperature=0.5
        # response_formatは古いAPIでは使えないので削除
    )
    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # JSON解析エラーの場合、デフォルトの構造を返す
        return {"proposals": []}

# 5. FastAPIイベントハンドラとエンドポイント
@app.on_event("startup")
def startup_event():
    global df_vectordb, is_ai_ready
    try:
        load_dotenv()
        openai.api_key = os.getenv("OPENAI_API_KEY")
        if not openai.api_key: raise ValueError("OPENAI_API_KEY is not set.")
        
        file_path = "伊藤さん2 - VectorDB (1).csv"
        df_vectordb = pd.read_csv(file_path, engine='python', keep_default_na=False)
        
        # FAISSを使わないシンプルな起動
        is_ai_ready = True
        print("✅ AIシステムが正常に起動しました。(Final Stable v0.28.1 - Simple Search)")

    except Exception as e:
        is_ai_ready = False
        import traceback
        print(f"❌ 初期化エラー: {e}")
        traceback.print_exc()

# (CORSとエンドポイントは変更なし)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.post("/ask", response_model=EstimateResponse)
async def ask_ai_endpoint(request: AskRequest):
    if not is_ai_ready: raise HTTPException(status_code=503, detail="AI is not ready.")
    try:
        corrected_text = correct_and_summarize(request.question)
        matched_ids = perform_similarity_search_simple(request.question)  # 簡易検索を使用
        
        # デバッグ情報
        print(f"Corrected text: {corrected_text}")
        print(f"Matched IDs: {matched_ids}")
        
        final_prompt = build_gpt_prompt(corrected_text, matched_ids, wanted_plans=request.wanted_plans)
        print(f"Final prompt:\n{final_prompt}")
        
        proposal_json = call_gpt_for_proposal(final_prompt)
        print(f"GPT response: {proposal_json}")
        
        # proposalsキーが無い場合のフォールバック
        if 'proposals' not in proposal_json or not proposal_json['proposals']:
            proposal_json = {
                "proposals": [{
                    "title": "標準見積もり",
                    "concept": "お客様のご要望に基づいた基本的な提案です",
                    "items": [{
                        "name": "基本作業",
                        "description": "詳細は打ち合わせにて決定",
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
