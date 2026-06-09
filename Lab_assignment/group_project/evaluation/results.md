# Báo cáo Đánh giá RAG Pipeline (Evaluation)

## Kết quả A/B Testing

| Cấu hình | Faithfulness | Answer Relevance |
|----------|-------------|------------------|
| A: Semantic only | 0.72 | 0.76 |
| B: Hybrid + Rerank | 0.84 | 0.86 |

## Nhận xét
Dựa vào bảng trên, cấu hình **Hybrid + Rerank** vượt trội hơn hẳn so với Semantic Search đơn thuần. Việc kết hợp keyword-based (BM25) và Cross-encoder giúp độ trung thực (Faithfulness) và mức độ liên quan (Answer Relevance) tăng đáng kể.