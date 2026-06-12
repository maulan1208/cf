# Hệ thống Recommendation (Recommender Systems)


## 1. Định nghĩa

Hệ thống Recommendation (gợi ý) là công cụ lọc thông tin, dự đoán sở thích người dùng để đề xuất sản phẩm, nội dung hoặc dịch vụ phù hợp. Được ứng dụng rộng rãi trong thương mại điện tử, streaming, mạng xã hội và nhiều lĩnh vực khác.

---

## 2. Các Phương pháp Chính

### 2.1 Collaborative Filtering (Lọc cộng tác)

Phương pháp phổ biến nhất, dự đoán sở thích dựa trên hành vi của nhóm người dùng tương tự.

| Kỹ thuật | Mô tả |
|---|---|
| **User-based CF** | Tìm người dùng "láng giềng" (k-NN) có sở thích giống nhau |
| **Item-based CF** | Tìm các item tương tự nhau dựa trên rating |
| **Matrix Factorization** | Phân tích ma trận user-item (SVD, ALS) → học đặc trưng ẩn |

**Ví dụ thực tế:**

```
Ma trận rating phim (1–5 sao, ? = chưa xem):

           Avengers  Titanic  Inception  Interstellar
  An           5        2        4            ?
  Bình         4        1        5            5
  Lan          2        5        1            2
  Mai          ?        4        2            1

→ User-based CF: An và Bình giống nhau (cùng thích action/sci-fi)
  → Gợi ý cho An: Interstellar (Bình chấm 5 sao)

→ Item-based CF: Inception và Interstellar thường được rating giống nhau
  → Người xem Inception → gợi ý Interstellar
```

> **Thực tế:** Amazon dùng Item-based CF ("Customers who bought X also bought Y"). Netflix dùng Matrix Factorization (SVD) đoạt giải Netflix Prize $1M năm 2009.

**Nhược điểm:**
- Sparsity: Ma trận user-item rất thưa
- Cold Start: Không xử lý được người dùng / item mới
- Tốn bộ nhớ khi dữ liệu lớn

---

### 2.2 Content-Based Filtering (Lọc theo nội dung)

Gợi ý dựa trên đặc tính của item so với lịch sử tương tác của người dùng.

**Kỹ thuật sử dụng:** TF-IDF, Cosine Similarity, Bayesian Classifier, Decision Tree, Clustering

**Ví dụ thực tế:**

```
User An thích: "Inception" (Sci-fi, Christopher Nolan, Rating 9.0)

Hệ thống vector hóa thuộc tính phim:
  Inception     → [sci-fi=1, thriller=1, nolan=1, rating=9.0]
  Interstellar  → [sci-fi=1, drama=1,   nolan=1, rating=8.6]
  Avengers      → [action=1, superhero=1,        rating=8.4]
  Tenet         → [sci-fi=1, thriller=1, nolan=1, rating=7.4]

Cosine similarity với Inception:
  Interstellar = 0.91  ✓ → GỢI Ý
  Tenet        = 0.87  ✓ → GỢI Ý
  Avengers     = 0.21  ✗

→ Gợi ý cho An: Interstellar, Tenet (không cần biết người khác xem gì)
```

> **Thực tế:** Spotify dùng audio features (tempo, energy, danceability) để gợi ý bài hát tương tự. News apps (Flipboard) dùng TF-IDF phân tích nội dung bài viết.

**Ưu điểm:**
- Không cần dữ liệu từ người dùng khác
- Giải quyết được cold start cho item mới

**Nhược điểm:**
- Bị giới hạn trong "bong bóng sở thích" (filter bubble)
- Khó gợi ý nội dung ngoài sở thích đã biết

---

### 2.3 Hybrid (Kết hợp)

Kết hợp Collaborative Filtering và Content-Based để tận dụng ưu điểm của cả hai.

- Cải thiện **41% độ chính xác** với người dùng có sở thích đa dạng
- Tăng **34% tỷ lệ giữ chân người dùng** trên các nền tảng số

**Ví dụ thực tế:**

```
Netflix Hybrid:
  Bước 1 (Content-Based): Lọc phim cùng thể loại/đạo diễn với lịch sử xem
  Bước 2 (CF): Xếp hạng bằng rating từ người dùng tương tự
  Bước 3 (Weighted): score_final = 0.4 × CF_score + 0.6 × CB_score

  → Người dùng mới (ít lịch sử): tăng trọng số CB (0.8)
  → Người dùng lâu năm (nhiều lịch sử): tăng trọng số CF (0.7)
```

> **Thực tế:** Spotify "Discover Weekly" kết hợp CF (playlist người dùng tương tự) + Content-Based (audio features) + NLP (phân tích blog nhạc). YouTube kết hợp CF + DNN + RL.

---

### 2.4 Knowledge-Based Recommendation (Gợi ý dựa trên tri thức)

Dùng tri thức miền (domain knowledge) thay vì dữ liệu lịch sử hành vi — phù hợp với các sản phẩm phức tạp, ít giao dịch (bất động sản, xe hơi, tài chính). Mang tính **hội thoại** (conversational): hệ thống hỏi để hiểu sở thích rồi mới gợi ý.

| Nhánh | Mô tả |
|---|---|
| **Constraint-based** | Định nghĩa tập ràng buộc (rules/constraints) để khớp yêu cầu người dùng với thuộc tính item |
| **Case-based** | Tìm item tương tự bằng cách tra cứu và điều chỉnh từ các "ca" (case) đã giải quyết trước đó |

**Ví dụ thực tế:**

```
Hệ thống gợi ý xe hơi (Constraint-based):
  Hệ thống hỏi:
    "Ngân sách của bạn?"       → User: < 600 triệu
    "Số chỗ ngồi?"             → User: 7 chỗ
    "Nhiên liệu?"              → User: Xăng hoặc Hybrid
    "Thương hiệu ưa thích?"    → User: Nhật Bản

  Constraint rules:
    price ≤ 600tr AND seats = 7 AND fuel IN [xăng, hybrid] AND origin = JP

  → Gợi ý: Toyota Innova Cross, Mitsubishi Xpander, Honda BR-V
  → Giải thích: "Phù hợp ngân sách, 7 chỗ, hybrid, thương hiệu Nhật"
```

**Ưu điểm:** Không cần dữ liệu lịch sử, giải thích được lý do gợi ý, không bị cold start.

> Nguồn: [Knowledge-based recommender systems: overview and research directions — Frontiers in Big Data 2024](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1304439/full)

---

### 2.5 Session-Based Recommendation (Gợi ý theo phiên)

Dự đoán item tiếp theo dựa trên **chuỗi hành vi trong một phiên tương tác ngắn**, không cần danh tính hay lịch sử dài hạn của người dùng. Đặc biệt hữu ích với người dùng ẩn danh.

**Kỹ thuật:** RNN/GRU, GNN (Graph Neural Network), Contrastive Learning

Framework **GECAF** (2024) kết hợp GNN + context-aware đạt **+2.03% P@20** và **+1.42% MRR@20** trên Tmall dataset.

**Ví dụ thực tế:**

```
Phiên mua sắm ẩn danh trên Shopee (không cần đăng nhập):
  Lịch sử phiên: Áo thun → Quần jean → Thắt lưng da → ?

  GRU4Rec học chuỗi:
    hidden_state = GRU(áo thun → quần jean → thắt lưng da)
    → dự đoán item tiếp theo: Giày da / Ví da / Tất

  → Gợi ý ngay trong phiên: "Giày da nam" (hoàn thiện outfit)

  Nếu phiên khác: Màn hình laptop → Bàn phím → Chuột → ?
  → Gợi ý: Pad chuột, Hub USB, Tai nghe
```

> Nguồn: [Graph-enhanced context aware framework for session-based recommendation — ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0925231224000389)

---

### 2.6 Knowledge Graph-Based Recommendation

Biểu diễn quan hệ user–item–attribute bằng **đồ thị tri thức (Knowledge Graph)**. Học embedding để nắm bắt quan hệ ngữ nghĩa phức tạp, giải quyết sparsity tốt hơn CF thuần túy.

**Kỹ thuật:** TransE, KGCN, GAT (Graph Attention Network), Transformer

- Giải quyết sparsity nhờ tích hợp thông tin phụ trợ (side information) qua các quan hệ trong đồ thị
- Ứng dụng mạnh trong giáo dục (e-learning), thương mại điện tử

**Ví dụ thực tế:**

```
Knowledge Graph cho thương mại điện tử:

  [iPhone 15] ──thuộc_hãng──→ [Apple]
  [iPhone 15] ──thuộc_loại──→ [Smartphone]
  [iPhone 15] ──tương_thích──→ [AirPods Pro]
  [iPhone 15] ──tương_thích──→ [MagSafe Charger]
  [Apple]     ──cũng_sản_xuất→ [MacBook Air]
  [User A]    ──đã_mua──────→ [iPhone 15]

  KGAT lan truyền embedding qua đồ thị:
    iPhone 15 → tương_thích → AirPods Pro → cùng_hãng → Apple Watch
    → Gợi ý cho User A: AirPods Pro, MagSafe Charger, Apple Watch

  Ưu điểm so với CF thuần: giải thích được "tại sao gợi ý"
    → "Vì bạn mua iPhone 15, sản phẩm này tương thích hoàn toàn"
```

> Nguồn: [A review of recommender systems based on knowledge graph embedding — Expert Systems with Applications 2024](https://dl.acm.org/doi/10.1016/j.eswa.2024.123876) · [RecKG — arXiv 2025](https://arxiv.org/html/2501.03598v1)

---

### 2.7 Federated Learning Recommendation (Gợi ý học liên kết)

Huấn luyện mô hình phân tán — **dữ liệu người dùng không rời khỏi thiết bị**. Mỗi client train cục bộ, chỉ gửi gradient/tham số lên server tổng hợp.

**Ưu điểm:** Bảo vệ privacy triệt để, tuân thủ GDPR, phù hợp với quy định dữ liệu nghiêm ngặt.

**Thách thức:** Dữ liệu không đồng nhất giữa các client (non-IID), chi phí truyền thông cao.

**Ví dụ thực tế:**

```
Bàn phím Gboard (Google) gợi ý từ tiếp theo:

  Điện thoại User A (Hà Nội):  train local → gradient_A
  Điện thoại User B (TP.HCM):  train local → gradient_B
  Điện thoại User C (Đà Nẵng): train local → gradient_C
                                      ↓
                         Server tổng hợp (FedAvg):
                         model_global = avg(gradient_A, B, C)
                                      ↓
                         Gửi model_global về từng máy

  → Lịch sử chat của từng người KHÔNG BAO GIỜ rời khỏi điện thoại
  → Mô hình ngày càng thông minh hơn mà không vi phạm privacy

Ứng dụng tương tự: Apple (Siri suggestions), Samsung Bixby
```

> Nguồn: [FedCL: Federated Contrastive Learning for Privacy-Preserving Recommendation — arXiv](https://arxiv.org/pdf/2204.09850) · [Adaptive course recommendation using federated learning — Nature 2025](https://www.nature.com/articles/s41598-025-26085-y.pdf)

---

### 2.8 Causal Inference / Debiasing (Suy luận nhân quả)

Phương pháp mới nhất — dùng **lý thuyết nhân quả** để loại bỏ các bias ẩn trong dữ liệu, đặc biệt là *popularity bias* (item phổ biến được gợi ý quá mức so với chất lượng thực).

| Kỹ thuật | Mô tả |
|---|---|
| **Backdoor Adjustment** | Loại bỏ confounding factor (bias phổ biến) khỏi quá trình học |
| **Counterfactual Reasoning** | Hỏi "nếu item không phổ biến, người dùng có thích không?" |
| **CIACC Framework** | Adversarial training kết hợp causal inference để xử lý long-tail |
| **CausalEPP** | Dùng time-series forecasting để tách popularity cá nhân vs. toàn cục |

**Ví dụ thực tế:**

```
Bài toán: Spotify có 100 triệu bài hát, nhưng top 1% bài chiếm 90% lượt nghe.
CF truyền thống → luôn gợi ý Sơn Tùng, BTS, Taylor Swift cho mọi người.

Causal Inference (Backdoor Adjustment):
  Câu hỏi nhân quả: "Nếu bài này KHÔNG nổi tiếng, user có nghe không?"

  Bài A: 10M lượt nghe (nổi tiếng) → user nghe vì nổi tiếng hay vì thực sự thích?
  Bài B: 1K lượt nghe (ít biết)   → user nghe → chắc chắn thực sự thích

  Sau debiasing:
    CF score(Bài A) = 4.2  →  Causal score = 4.2 - popularity_effect = 3.1
    CF score(Bài B) = 3.8  →  Causal score = 3.8 - 0.1 = 3.7

  → Bài B được xếp hạng cao hơn → khám phá nghệ sĩ underground
  → Người dùng thấy nội dung mới mẻ hơn, tăng thời gian ở lại app
```

> Nguồn: [Debiasing Recommendation with Personal Popularity — ACM Web Conference 2024](https://dl.acm.org/doi/10.1145/3589334.3645421) · [CIACC — ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S1568494625012414)

---

## 3. Phương pháp Deep Learning

| Mô hình | Ứng dụng |
|---|---|
| RBM, DBN, Autoencoder | Học đặc trưng tiềm ẩn từ dữ liệu thưa |
| CNN | Trích xuất đặc trưng từ ảnh, văn bản |
| RNN / LSTM | Recommendation tuần tự (sequential recommendation) |
| GNN (Graph Neural Network) | Mô hình hóa mối quan hệ phức tạp giữa user và item |
| Transformer / LLM | Hiểu ngữ nghĩa sâu, xử lý đa phương thức |
| Reinforcement Learning (RL) | Tối ưu hóa gợi ý dài hạn theo phản hồi thực |

---

## 4. Thách thức và Giải pháp

### 4.1 Cold Start Problem

Vấn đề phát sinh khi không có dữ liệu lịch sử của user hoặc item mới.

| Giải pháp | Mô tả |
|---|---|
| **CSRNet** | Kết hợp hierarchical clustering, transfer learning, Bi-GRU |
| **Contextual Bandit (LinUCB)** | Khám phá sở thích người dùng mới theo thời gian thực |
| **Thông tin nhân khẩu học** | Dùng tuổi, giới tính, vị trí khi chưa có lịch sử |
| **Active Learning** | Hỏi chủ động người dùng mới để thu thập sở thích |
| **Content-Based** | Dùng metadata của item ngay từ đầu |
| **Deep Learning + Side Info** | Tích hợp thông tin phụ (mô tả, ảnh) vào mô hình |

### 4.2 Các vấn đề khác

| Vấn đề | Giải pháp |
|---|---|
| **Sparsity** | Matrix Factorization, Autoencoder |
| **Scalability** | Approximate Nearest Neighbor (ANN), phân tán hóa |
| **Privacy** | Federated Learning, On-device inference |
| **Filter Bubble** | Thêm yếu tố đa dạng (diversity) vào kết quả |

---

## 5. Đánh giá Hiệu suất

### 5.1 Metrics đo lường lỗi dự đoán

| Metric | Ý nghĩa |
|---|---|
| **RMSE** | Đo sai số giữa rating dự đoán và thực tế; nhạy với outlier |
| **MAE** | Tương tự RMSE nhưng ít nhạy cảm với outlier hơn |

### 5.2 Metrics xếp hạng (Ranking Metrics)

| Metric | Ý nghĩa |
|---|---|
| **Precision@K** | Trong K gợi ý, bao nhiêu % là đúng |
| **Recall@K** | Trong tất cả item đúng, bao nhiêu % được đưa vào top K |
| **NDCG@K** | Đánh giá cả thứ tự — item đúng ở vị trí cao được điểm cao hơn |
| **MAP** | Trung bình precision qua nhiều mức recall |

> **Thực tế:** Các hệ thống như Netflix, Amazon kết hợp nhiều metrics. RMSE/MAE đảm bảo độ chính xác số học; MAP/NDCG phản ánh trải nghiệm xếp hạng thực tế của người dùng.

---

## 6. Xu hướng 2025 — LLM + Recommendation

Hướng nghiên cứu mới nhất kết hợp **Large Language Models** với hệ thống gợi ý:

- **LLM hiểu ngữ nghĩa sâu**: Xử lý text, ảnh, audio, video đa phương thức
- **On-device LLM**: Chạy mô hình nhỏ (Gemini Nano, LLaMA) trực tiếp trên thiết bị để bảo vệ privacy
- **Fine-tuning với LoRA**: Tinh chỉnh LLM hiệu quả cho từng domain (thương mại, giải trí, y tế)
- **LLM-powered Agents**: Tác tử tự động cá nhân hóa gợi ý theo ngữ cảnh phức tạp
- **Cross-domain Transfer Learning**: Chuyển kiến thức từ domain này sang domain khác
- **Knowledge Distillation**: Nén mô hình lớn thành mô hình nhỏ hơn, triển khai được trên edge

**Dự báo:** Giảm 65% chi phí tính toán, độ chính xác đạt 97% cho use case thông thường.

---

## 7. Ứng dụng Thực tế

| Nền tảng | Phương pháp chính |
|---|---|
| **Netflix** | Matrix Factorization + Deep Learning |
| **Spotify** | Collaborative Filtering + Audio features + NLP |
| **Amazon** | Item-based CF + Content-based Hybrid |
| **YouTube** | Deep Neural Networks + Reinforcement Learning |

---

## 8. Kiến trúc Tổng quan

```
┌─────────────────────────────────────────────────────┐
│                   Dữ liệu đầu vào                    │
│  (User behavior, Item metadata, Context, Ratings)    │
└──────────────────────┬──────────────────────────────┘
                       │
          ┌────────────▼────────────┐
          │   Feature Engineering   │
          │  (Embedding, TF-IDF...) │
          └────────────┬────────────┘
                       │
       ┌───────────────┼───────────────┐
       │               │               │
┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
│Collaborative│ │Content-Based│ │  Deep / LLM │
│  Filtering  │ │  Filtering  │ │   Models    │
└──────┬──────┘ └──────┬──────┘ └──────┬──────┘
       │               │               │
       └───────────────▼───────────────┘
                  ┌────┴────┐
                  │ Hybrid  │
                  │ Ranking │
                  └────┬────┘
                       │
          ┌────────────▼────────────┐
          │   Kết quả gợi ý @K      │
          │  (Precision, NDCG...)   │
          └─────────────────────────┘
```

---

## 9. Các Thuật toán Cụ thể và Ví dụ

### 9.1 Matrix Factorization

| Thuật toán | Đặc điểm | Ví dụ ứng dụng |
|---|---|---|
| **SVD** | Phân tích ma trận thành 2 latent factor matrix; chính xác cao trên dataset dày | Netflix Prize (2009) |
| **Funk-SVD** | Tối ưu chỉ trên rating đã có, hiệu quả với sparse data | MovieLens dataset |
| **SVD++** | Funk-SVD + implicit feedback (click, lịch sử xem) | Amazon product rating |
| **ALS** | Tối ưu luân phiên user/item, song song hóa tốt | Spark MLlib, Apple Music |
| **BPR** | Tối ưu ranking thay vì rating; pairwise loss | Tiktok video ranking |
| **NMF** | Ràng buộc giá trị không âm → đặc trưng dễ giải thích | Gợi ý tin tức |

**Ví dụ SVD:**
```
Ma trận R (User × Movie):        Sau SVD phân tích thành:
         M1  M2  M3  M4            R ≈ U × Σ × V^T
  User1 [ 5   3   ?   1 ]
  User2 [ 4   ?   4   1 ]    U: user latent factors  (sở thích ẩn)
  User3 [ 1   1   ?   5 ]    V: item latent factors  (đặc trưng ẩn)
  User4 [ ?   2   3   4 ]    Σ: độ quan trọng

→ Dự đoán R[User1, M3] = U[User1] · V[M3]^T = 4.2 ★
```

---

### 9.2 Embedding-based

| Thuật toán | Đặc điểm | Ví dụ ứng dụng |
|---|---|---|
| **Item2Vec** | Skip-gram trên chuỗi item; item = "từ", session = "câu" | Gợi ý sản phẩm thay thế |
| **Word2Vec** | Học embedding từ mô tả item bằng NLP | Gợi ý bài viết/tin tức |
| **DeepWalk** | Random walk trên đồ thị → học node embedding | Mạng xã hội (Pinterest) |

**Ví dụ Item2Vec:**
```
Lịch sử mua hàng các user:
  User A: [Áo thun, Quần jean, Giày thể thao]
  User B: [Áo sơ mi, Quần tây, Giày tây]
  User C: [Áo thun, Quần jean, Tất thể thao]

Skip-gram học: items thường xuất hiện cùng nhau → embedding gần nhau
  vector(Áo thun) ≈ vector(Quần jean)  [thường mua cùng]
  vector(Giày thể thao) ≈ vector(Tất thể thao)

→ User mua Áo thun + Quần jean → gợi ý Giày thể thao (embedding gần nhất)
```

---

### 9.3 Deep Learning

| Thuật toán | Đặc điểm | Ví dụ ứng dụng |
|---|---|---|
| **NCF** | MLP thay dot product → học tương tác phi tuyến | Pinterest, academic RS |
| **Wide & Deep** | Wide (memorization) + Deep (generalization) | Google Play Store |
| **DeepFM** | Factorization Machine + Deep Net; học feature interaction | Quảng cáo, CTR prediction |
| **DIN** | Attention động theo từng item đang xét | Alibaba Taobao |
| **SASRec** | Transformer decoder cho chuỗi; nhẹ, hiệu quả | Amazon, Steam games |
| **BERT4Rec** | Bidirectional Transformer; mask random item → dự đoán | Sequential RS |
| **LightGCN** | GCN đơn giản hóa; bỏ activation → hiệu quả hơn NGCF | Yelp, Amazon-Book |

**Ví dụ Wide & Deep (Google Play):**
```
Wide part (Memorization):
  Input: [user_installed_app × impression_app] → học co-occurrence trực tiếp
  "User đã cài Zalo thường cài Messenger" → memorize pattern cụ thể

Deep part (Generalization):
  Input: [age, gender, device, app_category, ...] → Embedding → MLP
  → Học pattern tổng quát: "user thích social app" → gợi ý mọi social app

Kết hợp: score = σ(W_wide · [a_wide, a_deep] + b)
→ Cân bằng giữa gợi ý chính xác theo lịch sử và khám phá app mới
```

---

### 9.4 Reinforcement Learning

| Thuật toán | Đặc điểm | Ví dụ ứng dụng |
|---|---|---|
| **DQN** | Deep Q-Network; học policy tối ưu dài hạn | Cold start, news RS |
| **Dueling DQN** | Tách value + advantage function; ổn định hơn | Large action space |
| **LinUCB** | Contextual Bandit; cân bằng explore/exploit | News recommendation |
| **REINFORCE** | Policy gradient; tối ưu trực tiếp reward | YouTube, Taobao |

**Ví dụ DQN cho News Recommendation:**
```
Môi trường: Trang tin tức
  State:    hồ sơ user + lịch sử đọc gần đây
  Action:   chọn bài báo nào để gợi ý (10.000+ bài)
  Reward:   +1 nếu user click, +3 nếu đọc >2 phút, 0 nếu bỏ qua

DQN học:
  Bước 1: Gợi ý bài "Chứng khoán tăng mạnh" → user bỏ qua → reward = 0
  Bước 2: Gợi ý bài "AI thay thế lập trình viên" → user đọc 5 phút → reward = 3
  Bước 3: Cập nhật Q-value → ưu tiên bài công nghệ cho user này

→ Khác CF: tối ưu engagement dài hạn, không chỉ next-item accuracy
```

---

### 9.5 Chọn thuật toán theo bài toán

| Tình huống | Thuật toán phù hợp | Lý do |
|---|---|---|
| Dataset nhỏ, explicit rating | SVD, Funk-SVD | Chính xác, ít dữ liệu cũng chạy tốt |
| Dataset lớn, implicit feedback | ALS, BPR, LightGCN | Scale tốt, tối ưu ranking |
| Gợi ý theo thứ tự thời gian | SASRec, BERT4Rec, GRU4Rec | Học chuỗi hành vi |
| Người dùng / item mới | LinUCB, Content-based, Knowledge-based | Không cần lịch sử |
| Quan hệ phức tạp, side info | KGAT, LightGCN + KG | Khai thác đồ thị tri thức |
| Yêu cầu privacy | Federated + ALS/GNN | Dữ liệu không rời thiết bị |
| Cần giải thích được | NMF, Knowledge-based, Causal | Interpretable |
| Scale cực lớn (tỷ item) | PinSage + FAISS | Approximate search |
| Tối ưu engagement dài hạn | DQN, REINFORCE | Reward dài hạn |

---

## Nguồn tham khảo

- [Evolution of recommendation systems in the age of AI — IJSRA 2025](https://journalijsra.com/sites/default/files/fulltext_pdf/IJSRA-2025-0061.pdf)
- [Combining Collaborative Filtering and Content Based Filtering — ResearchGate](https://www.researchgate.net/publication/383811239_Combining_Collaborative_Filtering_and_Content_Based_Filtering_for_Recommendation_Systems)
- [A collaborative filtering recommender systems: Survey — ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925231224014899)
- [In-depth survey: deep learning in recommender systems — Springer](https://link.springer.com/article/10.1007/s00521-024-10866-z)
- [A Comprehensive Survey of Evaluation Techniques — arXiv](https://arxiv.org/html/2312.16015v2)
- [10 metrics to evaluate recommender systems — Evidently AI](https://www.evidentlyai.com/ranking-metrics/evaluating-recommender-systems)
- [Cold Start Problem solutions — Tredence](https://www.tredence.com/blog/solving-the-cold-start-problem-in-collaborative-recommender-systems)
- [Large Language Model Enhanced Recommender Systems: A Survey — arXiv](https://arxiv.org/abs/2412.13432)
- [Improving Recommendation Systems in the Age of LLMs — Eugene Yan](https://eugeneyan.com/writing/recsys-llm/)
- [Knowledge-based recommender systems: overview and research directions — Frontiers in Big Data 2024](https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2024.1304439/full)
- [Graph-enhanced context aware framework for session-based recommendation — ScienceDirect 2024](https://www.sciencedirect.com/science/article/abs/pii/S0925231224000389)
- [A review of recommender systems based on knowledge graph embedding — Expert Systems with Applications 2024](https://dl.acm.org/doi/10.1016/j.eswa.2024.123876)
- [RecKG: Knowledge Graph for Recommender Systems — arXiv 2025](https://arxiv.org/html/2501.03598v1)
- [FedCL: Federated Contrastive Learning for Privacy-Preserving Recommendation — arXiv](https://arxiv.org/pdf/2204.09850)
- [Adaptive course recommendation using federated learning — Nature 2025](https://www.nature.com/articles/s41598-025-26085-y.pdf)
- [Debiasing Recommendation with Personal Popularity — ACM Web Conference 2024](https://dl.acm.org/doi/10.1145/3589334.3645421)
- [Causal inference and attribute correlation consistency (CIACC) — ScienceDirect 2025](https://www.sciencedirect.com/science/article/abs/pii/S1568494625012414)
