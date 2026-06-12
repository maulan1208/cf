# News Recommendation System
---

## 1. Từ Recommendation → News Recommendation

Các khái niệm trong recommendation tổng quát đều áp dụng được cho news, nhưng **news có đặc thù riêng** buộc phải điều chỉnh:

| Khái niệm tổng quát | Trong News Recommendation |
|---|---|
| Item | Bài báo (article) |
| User rating (explicit) | Hầu như không có — thay bằng implicit |
| Implicit feedback | Click, thời gian đọc, chia sẻ, cuộn trang |
| Item metadata | Tiêu đề, nội dung, chủ đề, tác giả, thời gian đăng |
| User history | Lịch sử bài đã đọc (ngắn hạn + dài hạn) |
| Cold start (item) | **Xảy ra liên tục** — hàng nghìn bài mới mỗi ngày |
| Popularity | Trending (thay đổi theo giờ, không theo tháng) |

---

## 2. Đặc thù của News — Tại sao khó hơn

### 2.1 Time Decay (Tin tức có hạn dùng)

```
Bài đăng lúc 08:00 → hot lúc 09:00 → cũ lúc 14:00 → vô nghĩa lúc 08:00 hôm sau

Phim "Inception" (2010) vẫn được gợi ý năm 2025 ✓
Bài "Kết quả bầu cử hôm qua" đăng 2 ngày trước   ✗
```

**Giải pháp:** Nhân score với hàm suy giảm thời gian:

```
recency_score = e^(-λ × age_in_hours)
  λ = 0.1 → bài 24h còn ~9% điểm ban đầu
  λ = 0.5 → bài 6h còn ~5% điểm ban đầu
```

### 2.2 Item Cold Start liên tục

```
E-commerce: item mới thêm mỗi tuần → cold start thỉnh thoảng
News:       bài mới đăng mỗi phút  → cold start LIÊN TỤC

→ CF thuần túy thất bại hoàn toàn cho bài mới
→ Phải dùng Content-Based ngay từ đầu
```

### 2.3 Implicit Feedback phức tạp

```
Người dùng KHÔNG chấm sao bài báo như chấm phim Netflix.
Hành vi thực tế phải suy luận:

  Click + đọc 5 giây  → không thích (clickbait)      rating ≈ 0
  Click + đọc 3 phút  → thích                         rating ≈ 3
  Click + chia sẻ     → rất thích                     rating ≈ 5
  Bỏ qua (no click)   → không quan tâm / đã biết      rating ≈ 0
  Scroll nhanh        → thấy nhưng không thích         rating ≈ -0.5
```

### 2.4 Sở thích thay đổi theo thời gian

```
Tháng 6:  User quan tâm World Cup → gợi ý bóng đá ✓
Tháng 8:  World Cup kết thúc      → gợi ý bóng đá ✗

→ Cần mô hình hóa cả sở thích ngắn hạn (tuần này) và dài hạn (năm nay)
```

---

## 3. Áp dụng các Phương pháp Recommendation vào News

### 3.1 Collaborative Filtering (CF) — Hạn chế và Cách tận dụng

#### Tại sao CF thuần túy thất bại với news

```
Bình thường (phim):
  User A: xem Inception → Rating 5★
  User B: xem Inception → Rating 5★ + xem Interstellar → Rating 5★
  → CF gợi ý Interstellar cho A  ✓  (Inception có hàng nghìn rating)

Với news:
  Bài "Lũ lụt Hà Nội 8h sáng" đăng lúc 08:00
  → Lúc 08:05: chỉ 3 người đọc → CF không đủ data
  → Lúc 14:00: 50.000 người đọc → data đủ nhưng tin đã cũ
  → CF bỏ lỡ hoàn toàn thời điểm vàng của bài
```

#### CF hoạt động được ở cấp độ nào trong News

CF vẫn có ích nếu dùng ở **cấp độ chủ đề/category** thay vì bài cụ thể:

```
Ma trận User × Category (implicit feedback — số bài đã đọc):

              KinhTe  TheThao  CongNghe  GiaiTri  ChinhTri
  An              12       45         8        2        1
  Bình             3        2        50       10        5
  Lan             20        5        15        1       30
  Mai              1       40         3       20        2

ALS factorize → học latent preference:
  An   → [thể_thao_fan, kinh_tế_mild]
  Bình → [công_nghệ_fan, giải_trí_mild]

→ Gợi ý cho An: ưu tiên bài TheThao + KinhTe
→ Gợi ý cho Bình: ưu tiên bài CongNghe + GiaiTri
```

#### Implicit Feedback trong News — Cách quy đổi

| Hành vi | Weight | Lý do |
|---|---|---|
| Click + đọc > 3 phút | 1.0 | Rõ ràng thích |
| Click + đọc 30s–3 phút | 0.5 | Có hứng thú |
| Click + thoát < 30s | 0.0 | Clickbait, không tính |
| Chia sẻ | 2.0 | Tín hiệu mạnh nhất |
| Comment | 1.5 | Quan tâm sâu |
| Scroll qua không click | -0.1 | Nhẹ không thích |

```python
# Quy đổi implicit feedback sang rating
def compute_rating(click, read_time_sec, shared, commented):
    if not click:
        return -0.1           # scroll qua
    if read_time_sec < 30:
        return 0.0            # clickbait
    score = min(read_time_sec / 180, 1.0) * 0.5   # read time (tối đa 0.5)
    score += 2.0 if shared else 0
    score += 1.5 if commented else 0
    return min(score, 3.0)    # cap tại 3.0
```

#### Khi nào dùng CF trong News

- Gợi ý **chuyên mục** cho user mới đăng ký (biết họ thích mảng nào)
- Gợi ý **tác giả** yêu thích (user A và B đều follow cùng phóng viên)
- Gợi ý **bài cũ nhưng còn giá trị** (evergreen content) trong sidebar
- Kết hợp với Content-Based trong hybrid pipeline

---

### 3.2 Content-Based Filtering — Xương sống của News RS

Đây là phương pháp **không thể thiếu** vì là duy nhất xử lý được cold start liên tục.

#### Bước 1 — Xây dựng News Encoder (biểu diễn bài báo)

```
Bài báo mới đăng:
  Tiêu đề:  "Giá xăng tăng mạnh, người dân than thở"
  Nội dung: "Từ 15h hôm nay, giá xăng E5 RON95 tăng 500đ/lít..."
  Chủ đề:   Kinh tế / Đời sống
  Thực thể: [xăng, RON95, Bộ Công Thương, Việt Nam]

Pipeline xử lý:

  ┌─ TF-IDF (nhanh, nhẹ) ────────────────────────────────┐
  │  "giá xăng tăng" → vector thưa [0, 0, 0.8, 0, 0.6, ...]  │
  │  Ưu: real-time, không cần GPU                          │
  │  Nhược: không hiểu ngữ nghĩa ("tăng" ≠ "leo thang")  │
  └───────────────────────────────────────────────────────┘

  ┌─ BERT (chính xác, hiểu ngữ nghĩa) ───────────────────┐
  │  "giá xăng tăng" → dense vector [0.23, -0.45, 0.81, ...]  │
  │  "giá nhiên liệu leo thang" → vector gần tương đương   │
  │  Ưu: hiểu nghĩa, đồng nghĩa, ngữ cảnh                │
  │  Nhược: cần GPU, chậm hơn (~50ms/bài)                 │
  └───────────────────────────────────────────────────────┘

  ┌─ NER — Named Entity Recognition ─────────────────────┐
  │  Trích: {entity: "xăng E5 RON95", type: "PRODUCT"}   │
  │         {entity: "Bộ Công Thương", type: "ORG"}       │
  │         {entity: "Việt Nam", type: "GPE"}             │
  │  → Dùng để liên kết Knowledge Graph                   │
  └───────────────────────────────────────────────────────┘

  → Kết hợp: news_vector = concat(BERT_emb, category_onehot, entity_emb)
```

#### Bước 2 — Xây dựng User Profile từ lịch sử

```
Lịch sử đọc của An (7 ngày qua):

  Bài 1: "VN tăng trưởng GDP 7%"        → vec_1, weight=1.0 (đọc 4 phút)
  Bài 2: "Lạm phát toàn cầu hạ nhiệt"   → vec_2, weight=0.8 (đọc 2.5 phút)
  Bài 3: "Giá xăng tăng mạnh"           → vec_3, weight=2.0 (chia sẻ)
  Bài 4: "Kết quả bóng đá hôm nay"      → vec_4, weight=0.0 (thoát <30s)

User Profile = weighted average của news vectors:
  user_vec = (1.0×vec_1 + 0.8×vec_2 + 2.0×vec_3 + 0.0×vec_4) / (1.0+0.8+2.0)
           = trung bình có trọng số → thiên về chủ đề kinh tế/xăng dầu

→ Thêm time decay: bài đọc hôm qua weight × 0.9, tuần trước × 0.5
```

#### Bước 3 — Tính Similarity và Gợi ý

```
Bài ứng viên mới: "OPEC cắt giảm sản lượng, ảnh hưởng giá xăng VN"
  → candidate_vec = BERT_encode(tiêu đề + nội dung)

Cosine similarity:
  sim = dot(user_vec, candidate_vec) / (|user_vec| × |candidate_vec|)
      = 0.87  ← rất cao

So sánh với các bài khác:
  "Giải Oscar 2025":           sim = 0.12  ✗
  "Thị trường chứng khoán VN": sim = 0.71  ✓
  "OPEC cắt giảm sản lượng":   sim = 0.87  ✓✓ → ưu tiên gợi ý
```

#### TF-IDF vs BERT — Khi nào dùng cái nào

| Tiêu chí | TF-IDF | BERT |
|---|---|---|
| Tốc độ encode | <1ms | 30–100ms |
| Cần GPU | Không | Có (hoặc dùng bản nhỏ) |
| Hiểu đồng nghĩa | Không | Có |
| Xử lý real-time | Dễ | Cần tối ưu |
| Phù hợp | Candidate generation nhanh | Ranking chính xác |
| Ví dụ dùng | Tầng 1 (recall) | Tầng 2 (ranking) |

---

### 3.3 Session-Based — Gợi ý cho User Ẩn danh

Hơn **70% traffic** báo điện tử đến từ người dùng chưa đăng nhập — session-based là phương pháp duy nhất phục vụ nhóm này.

#### GRU4Rec — Cơ chế hoạt động

```
Phiên đọc của user ẩn danh:

  t=0: click "Lũ lụt Hà Nội"        → h0 = GRU(embed("lũ lụt Hà Nội"))
  t=1: click "Sơ tán dân vùng ngập" → h1 = GRU(embed("sơ tán"), h0)
  t=2: click "Thiệt hại ước tính"   → h2 = GRU(embed("thiệt hại"), h1)
  t=3: ? → dự đoán next item

GRU học: mỗi hidden state h_t tóm tắt ngữ cảnh phiên đến thời điểm t
  h0: [thiên_tai=high, Hà_Nội=high]
  h1: [thiên_tai=high, cứu_hộ=medium, Hà_Nội=high]
  h2: [thiên_tai=high, thiệt_hại=high, cứu_hộ=high]

→ Gợi ý t=3: "Hỗ trợ người dân vùng lũ", "Dự báo mưa Hà Nội"
             "Chỉ đạo của Thủ tướng về ứng phó lũ"
```

#### SASRec — Transformer cho Session News

```
So với GRU4Rec, SASRec dùng Self-Attention thay GRU:

Input: [bài_1, bài_2, bài_3] (sequence trong phiên)

         bài_1    bài_2    bài_3
          ↓        ↓        ↓
       [embed]  [embed]  [embed]
          \        |        /
           Self-Attention
          /        |        \
       query_1  query_2  query_3
                              ↓
                     dự đoán bài_4

Ưu điểm so với GRU:
  GRU:    học tuần tự, bài đầu phiên "bị quên" dần
  SASRec: attention thấy toàn bộ phiên cùng lúc
          → bài đầu phiên vẫn ảnh hưởng đến dự đoán cuối phiên
```

#### Xác định ranh giới phiên (Session Boundary)

```
Cách 1 — Timeout:
  Nếu user không có hành vi > 30 phút → kết thúc phiên, bắt đầu phiên mới

Cách 2 — Chủ đề thay đổi đột ngột:
  Đọc 5 bài về "lũ lụt" → đột ngột click "Kết quả xổ số"
  → Cosine sim giữa bài mới và trung bình phiên < 0.3
  → Phát hiện chủ đề shift → reset context (hoặc tách phiên)

Cách 3 — Kết hợp: timeout + topic shift
```

#### Hybrid Session — Kết hợp Session + Content

```
user ẩn danh, phiên hiện tại: [Kinh tế A, Kinh tế B, Kinh tế C]

score(bài_X) = α × GRU4Rec_score(bài_X | phiên)     ← ngữ cảnh phiên
             + β × CB_score(bài_X | avg_session_vec)  ← content similarity
             + γ × recency_score(bài_X)               ← bài còn mới

α=0.5, β=0.3, γ=0.2 (điều chỉnh qua A/B test)
```

> Nguồn: [Hybrid Session-based News Recommendation using RNN — arXiv](https://arxiv.org/pdf/2006.13063) · [Diversification in Session-based News RS — arXiv](https://arxiv.org/pdf/2102.03265)

---

### 3.4 Knowledge Graph — Liên kết Sự kiện và Thực thể

#### Xây dựng News Knowledge Graph

```
Từ các bài báo, trích xuất thực thể và quan hệ:

Bài: "Thủ tướng Phạm Minh Chính chỉ đạo ứng phó bão số 3"

NER trích xuất:
  PERSON:  Phạm Minh Chính
  EVENT:   Bão số 3 (Yagi)
  ACTION:  chỉ đạo ứng phó
  ORG:     Chính phủ Việt Nam

→ Thêm vào Knowledge Graph:
  (Phạm Minh Chính) ──chỉ_đạo──→ (ứng phó bão số 3)
  (Bão số 3)        ──gây_ra───→ (Lũ lụt miền Bắc)
  (Lũ lụt miền Bắc) ──ảnh_hưởng→ (Nông nghiệp VN)
  (Bão số 3)        ──xuất_phát→ (Biển Đông)
```

#### KGAT — Lan truyền Embedding qua Đồ thị

```
User đọc: "Bão số 3 đổ bộ Hải Phòng"

KGAT lan truyền 2 bước:
  Bước 1 (hop-1): các node trực tiếp liên quan đến "Bão số 3"
    → Lũ lụt miền Bắc, Thiệt hại nông nghiệp, Sơ tán dân

  Bước 2 (hop-2): các node liên quan đến hop-1
    → Hỗ trợ thiên tai, Tái thiết sau lũ, Dự báo thời tiết

  Attention weight: ưu tiên relation "gây_ra" > "liên_quan"

→ Gợi ý theo độ gần trên đồ thị:
  1. "Thiệt hại lũ lụt ước 5.000 tỷ"     (hop-1, weight cao)
  2. "Chính phủ hỗ trợ vùng lũ"          (hop-2, weight trung bình)
  3. "Dự báo mưa lớn tiếp tục"           (hop-2, weight trung bình)
  4. "Giá lúa gạo tăng sau lũ"           (hop-2, weight thấp hơn)

→ Tốt hơn keyword matching:
  keyword "bão":  chỉ tìm bài có chữ "bão"
  KG:             hiểu "lũ lụt" và "thiệt hại nông nghiệp"
                  đều liên quan đến "bão số 3"
```

#### Thực thể đặc thù trong News KG

| Loại thực thể | Ví dụ | Quan hệ phổ biến |
|---|---|---|
| PERSON | Chính trị gia, CEO, VĐV | phát_biểu, chỉ_đạo, liên_quan |
| ORG | Chính phủ, Công ty, CLB | ban_hành, tham_gia, sở_hữu |
| EVENT | Bầu cử, World Cup, IPO | gây_ra, diễn_ra_tại, có_kết_quả |
| GPE | Quốc gia, thành phố | nằm_trong, bị_ảnh_hưởng |
| PRODUCT | Xăng RON95, iPhone 16 | tăng_giá, ra_mắt, bị_thu_hồi |

---

### 3.5 Reinforcement Learning — Tối ưu Engagement Dài hạn

#### Định nghĩa bài toán RL cho News

```
Môi trường (Environment): Trang báo điện tử

  State (s_t):
    - User profile vector (từ lịch sử đọc)
    - Lịch sử K bài vừa gợi ý
    - Thời điểm trong ngày (sáng/chiều/tối)
    - Device (mobile/desktop)
    - Bài vừa đọc xong

  Action (a_t):
    - Chọn bài nào để gợi ý tiếp theo
    - (10.000+ bài → dùng action embedding, không enumerate hết)

  Reward (r_t):
    r_t = 0.0   nếu không click
    r_t = 0.3   nếu click nhưng đọc < 30s  (clickbait penalty)
    r_t = 1.0   nếu đọc 1–3 phút
    r_t = 2.0   nếu đọc > 3 phút
    r_t = 3.0   nếu chia sẻ
    r_t = 5.0   nếu quay lại app trong 24h tiếp theo (long-term signal)

  Goal: tối đa hóa tổng reward tích lũy: Σ γ^t × r_t  (γ=0.95)
```

#### DQN cho News — Vòng lặp Học

```
Episode = 1 phiên đọc của user

  t=0: State = {user_profile, "vừa vào app"}
       Agent chọn Action = gợi ý bài A (Kinh tế)
       User click, đọc 4 phút → Reward = 2.0
       Next state = {user_profile + đọc kinh tế, bài A}

  t=1: Agent chọn Action = gợi ý bài B (Kinh tế khác)
       User click, đọc 30s → Reward = 0.3  (ít hứng thú hơn)
       → DQN học: không nên gợi ý liên tiếp 2 bài cùng chủ đề quá sát

  t=2: Agent chọn Action = gợi ý bài C (Thể thao)
       User chia sẻ → Reward = 3.0
       → DQN học: sau kinh tế, user này thích thể thao

  t=N: User thoát app
       → Nếu user quay lại trong 24h: bonus reward = 5.0
```

#### LinUCB — Cho User Mới (Cold Start)

```
User mới chưa có lịch sử → RL phải khám phá sở thích:

LinUCB (Upper Confidence Bound):
  score(bài_i) = θ^T × x_i  +  α × √(x_i^T A_i^{-1} x_i)
                 ↑ exploitation   ↑ exploration bonus

  x_i: feature vector của bài i (category, entity, recency)
  θ:   tham số học được từ feedback
  α:   hệ số cân bằng explore/exploit

Vòng lặp:
  Gợi ý bài → user click/bỏ qua → cập nhật θ → gợi ý lại

Ví dụ với user mới:
  Lần 1: thử gợi ý Kinh tế  → click, đọc 3p  → θ[kinh_tế] tăng
  Lần 2: thử gợi ý TheThao  → bỏ qua         → θ[thể_thao] giảm
  Lần 3: thử gợi ý CongNghe → click, chia sẻ  → θ[công_nghệ] tăng mạnh
  Lần 4–∞: exploit Kinh tế + CongNghe, vẫn explore nhẹ các chủ đề khác
```

#### So sánh RL vs. Supervised Learning cho News

| Tiêu chí | Supervised (NRMS, NAML) | Reinforcement Learning |
|---|---|---|
| Tối ưu | Accuracy trên tập test cố định | Reward tích lũy dài hạn |
| Feedback | Offline (log cũ) | Online (real-time) |
| Chống clickbait | Kém (nếu data có clickbait) | Tốt (reward penalty) |
| Thời gian train | Nhanh (offline) | Chậm (cần nhiều interaction) |
| Độ phức tạp | Trung bình | Rất cao |
| Phù hợp khi | Có nhiều log offline | Muốn tối ưu UX dài hạn |

> Nguồn: [Deep RL Recommendation System — Shaped.ai](https://www.shaped.ai/blog/deep-reinforcement-learning-for-recommender-systems--a-survey) · [LinUCB for Cold Start — arXiv](https://arxiv.org/pdf/1405.7544)

---

## 4. Các Mô hình Chuyên biệt cho News

### 4.1 NAML — Neural Attentive Multi-view Learning (2019)

Học biểu diễn bài báo từ **nhiều góc nhìn** (multi-view):

```
Bài báo → [Tiêu đề] ──CNN + Attention──┐
        → [Nội dung] ──CNN + Attention──┼──→ News Embedding
        → [Chủ đề]   ──Embedding────────┘

User History → NAML(bài 1, bài 2, ...) → User Embedding

Score = dot(User Embedding, News Embedding)
```

### 4.2 LSTUR — Long-Short Term User Representation (2019)

Mô hình hóa **sở thích dài hạn và ngắn hạn** song song:

```
Long-term:   User ID → Embedding (học từ toàn bộ lịch sử)
Short-term:  GRU(bài đọc gần đây) → Hidden state

User Repr = concat(long_term, short_term)

Ví dụ:
  Long-term:  user thích [Kinh tế, Thể thao] (ổn định)
  Short-term: tuần này đọc nhiều về [World Cup] (tạm thời)
  → Gợi ý: bài kinh tế + bài World Cup
```

### 4.3 NRMS — Neural News Recommendation with Multi-Head Self-Attention (2019)

Dùng **Transformer self-attention** cho cả news encoding và user modeling:

```
News Encoder:
  Tiêu đề: [w1, w2, ..., wn]
  → Multi-head Self-Attention → học quan hệ giữa các từ
  → Additive Attention → news vector

User Encoder:
  [bài 1, bài 2, ..., bài k] (news vectors)
  → Multi-head Self-Attention → học quan hệ giữa các bài đã đọc
  → Additive Attention → user vector

Score = dot(user_vector, candidate_news_vector)
```

**So sánh hiệu suất trên MIND dataset:**

| Mô hình | AUC | MRR | nDCG@5 | nDCG@10 |
|---|---|---|---|---|
| NAML | 0.6357 | 0.2967 | 0.3216 | 0.3825 |
| LSTUR | 0.6481 | 0.3084 | 0.3350 | 0.3939 |
| NRMS | 0.6322 | 0.2977 | 0.3263 | 0.3836 |
| Co-NAML-LSTUR | **0.6571** | **0.3119** | **0.3397** | **0.3979** |

### 4.4 UNBERT — User-News Matching BERT (2021)

Dùng **BERT pre-trained** để hiểu ngữ nghĩa sâu:

```
Input: [CLS] tiêu đề bài ứng viên [SEP] tiêu đề bài đã đọc 1 ... [SEP]

BERT xử lý đồng thời:
  - Word-level matching: từ nào trong bài ứng viên khớp với lịch sử
  - News-level matching: bài ứng viên có phù hợp sở thích tổng thể không

Ưu điểm: hiểu được "Mưa lớn ở Hà Nội" ~ "Ngập lụt thủ đô"
          (không cần trùng từ khóa)
```

### 4.5 Mô hình LLM-based (2024–2025)

```
Prompt cho LLM:
  "Người dùng đã đọc:
   - 'VN tăng trưởng GDP 7% quý 3'
   - 'Xuất khẩu gạo đạt kỷ lục'
   - 'FDI vào Việt Nam tăng mạnh'

   Trong các bài sau, bài nào phù hợp nhất?
   A. 'Lạm phát Mỹ hạ nhiệt'
   B. 'Đầu tư công thúc đẩy tăng trưởng VN'
   C. 'Kết quả bóng đá hôm nay'"

→ LLM hiểu ngữ cảnh → chọn B, giải thích lý do
```

---

## 5. Pipeline News Recommendation thực tế

```
┌──────────────────────────────────────────────────────────┐
│                    Thu thập dữ liệu                       │
│   Crawler bài mới → NLP pipeline → Index vào DB          │
└─────────────────────────┬────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
   ┌──────▼──────┐ ┌──────▼──────┐ ┌──────▼──────┐
   │  News       │ │   User      │ │  Context    │
   │  Encoder    │ │  Encoder    │ │  Features   │
   │  (BERT/CNN) │ │(LSTUR/NRMS) │ │(time, geo)  │
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          └───────────────▼───────────────┘
                    ┌─────┴─────┐
                    │  Ranking  │
                    │  Model    │
                    └─────┬─────┘
                          │
             ┌────────────┼────────────┐
             │            │            │
       Relevance    Recency       Diversity
        Score       Score          Score
             │            │            │
             └────────────▼────────────┘
                   Final Score
                  = α×relevance + β×recency + γ×diversity
                          │
                   Top-K Articles
```

---

## 6. Thách thức đặc thù và Giải pháp

### 6.1 Filter Bubble (Bong bóng thông tin)

```
Vấn đề: User chỉ đọc tin ủng hộ quan điểm của mình
         → RS tiếp tục gợi ý → tư duy ngày càng cực đoan

Giải pháp:
  Diversity regularization:
    score_final = relevance - λ × similarity_to_already_recommended

  Ví dụ: đã gợi ý 5 bài về "chính sách A" → hạ điểm bài thứ 6 cùng chủ đề
         → tự động thêm 1 bài góc nhìn khác
```

### 6.2 Clickbait vs. Quality

```
Clickbait: CTR cao nhưng reading time thấp, bounce rate cao

Multi-objective optimization:
  reward = CTR × 0.2 + ReadTime × 0.4 + ShareRate × 0.2 + ReturnRate × 0.2

→ Bài chất lượng thật sự mới có reward cao
```

### 6.3 Breaking News (Tin nóng)

```
Sự kiện bất ngờ xảy ra lúc 10:00:
  10:01 → bài đầu tiên đăng, 0 người đọc
  10:05 → Google Trends tăng đột biến về keyword này

Giải pháp Trending Detection:
  trending_score = (search_volume_now - search_volume_1h_ago) / baseline
  → Bài về keyword trending được boost lên feed ngay
     dù chưa có implicit feedback
```

---

## 7. Dataset Chuẩn để Nghiên cứu

| Dataset | Nguồn | Quy mô | Đặc điểm |
|---|---|---|---|
| **MIND** | Microsoft News | 160K bài, 1M users | Benchmark chuẩn nhất, có train/dev/test |
| **Adressa** | Na-uy | 2.7M lượt đọc | Có dữ liệu thời gian đọc chi tiết |
| **DBLP** | Báo khoa học | - | Cho academic RS |
| **Globo** | Brazil | 1M sessions | Session-based news |

**MIND Dataset structure:**
```
news.tsv:     NewsID | Category | SubCategory | Title | Abstract | URL | ...
behaviors.tsv: ImpressionID | UserID | Time | History | Impressions

Impressions: "N1-1 N2-0 N3-1 N4-0"
  → N1 clicked (1), N2 not clicked (0), N3 clicked (1)...
```

---

## 8. Đánh giá News Recommendation

| Metric | Ý nghĩa trong News |
|---|---|
| **AUC** | Phân biệt bài user sẽ click vs. không click |
| **MRR** | Bài đúng đầu tiên xuất hiện ở vị trí nào |
| **nDCG@K** | Chất lượng xếp hạng top K bài |
| **Intra-list Diversity** | Đo độ đa dạng trong K bài gợi ý |
| **Serendipity** | Mức độ bất ngờ tích cực (khám phá ngoài vùng comfort) |

---

## 9. So sánh News RS vs. General RS

| Tiêu chí | General RS (phim, sản phẩm) | News RS |
|---|---|---|
| Vòng đời item | Dài (năm, thập kỷ) | Ngắn (giờ, ngày) |
| Cold start | Thỉnh thoảng | Liên tục (hàng nghìn bài/ngày) |
| Explicit feedback | Phổ biến (rating, review) | Gần như không có |
| Sở thích user | Tương đối ổn định | Thay đổi theo sự kiện |
| Mục tiêu | Accuracy | Accuracy + Freshness + Diversity |
| Phương pháp chủ đạo | CF + MF | Content-Based (BERT) + RL |

---

## 10. Chọn Phương pháp — Hướng dẫn Chi tiết

### 10.1 Các Tiêu chí Quyết định

Trước khi chọn phương pháp, cần trả lời **5 câu hỏi** sau:

| # | Câu hỏi | Ảnh hưởng đến lựa chọn |
|---|---|---|
| 1 | Lượng dữ liệu lịch sử hiện có? | Ít → Content-Based; nhiều → CF + MF |
| 2 | Tần suất bài mới xuất hiện? | Cao → Content-Based bắt buộc |
| 3 | Yêu cầu latency? | <50ms → ANN index; <500ms → ranking model |
| 4 | User có đăng nhập không? | Không → Session-based / Popularity |
| 5 | Cần giải thích gợi ý không? | Có → Knowledge-based / Content-Based |

---

### 10.2 Kiến trúc 2 Tầng (Industry Standard)

Hệ thống thực tế **không chạy 1 mô hình duy nhất** — luôn chia thành 2 giai đoạn:

```
Toàn bộ kho bài (10.000+ bài)
         │
         ▼
┌─────────────────────────────────────────┐
│  TẦNG 1: CANDIDATE GENERATION (Recall) │
│                                         │
│  Mục tiêu: lọc nhanh còn ~200-500 bài  │
│  Yêu cầu: tốc độ cao, recall cao        │
│  Mô hình: đơn giản, nhẹ                 │
│                                         │
│  • Two-Tower Model (ANN search)         │
│  • BM25 keyword matching                │
│  • Popularity + Recency filter          │
│  • Collaborative Filtering (ALS)        │
└──────────────────┬──────────────────────┘
                   │ ~200-500 bài ứng viên
                   ▼
┌─────────────────────────────────────────┐
│  TẦNG 2: RANKING (Re-rank)             │
│                                         │
│  Mục tiêu: xếp hạng chính xác top-K    │
│  Yêu cầu: chính xác cao                 │
│  Mô hình: phức tạp, tốn tài nguyên      │
│                                         │
│  • NRMS / NAML / LSTUR                  │
│  • LightGBM + feature engineering       │
│  • Deep Learning (BERT + MLP)           │
│  • RL-based re-ranker                   │
└──────────────────┬──────────────────────┘
                   │ Top 10-20 bài
                   ▼
              Feed người dùng
```

**Lý do tách 2 tầng:**
- Tầng 1 chạy trên **toàn bộ kho** → phải cực nhanh (ms)
- Tầng 2 chỉ chạy trên **~500 bài** → có thể dùng mô hình nặng

---

### 10.3 Decision Tree — Chọn phương pháp theo tình huống

```
START
  │
  ├─ User chưa đăng nhập / không có lịch sử?
  │     ├─ YES → Popularity-based + Trending
  │     │         + Content-Based từ metadata bài
  │     │
  │     └─ NO (có lịch sử) ──────────────────────────────┐
  │                                                        │
  ├─ Bài báo vừa đăng (<1 giờ)?                          │
  │     ├─ YES → Content-Based (BERT embedding tiêu đề)   │
  │     │         + Trending score boost                   │
  │     │                                                  │
  │     └─ NO (bài cũ hơn) ──────────────────┐            │
  │                                           │            │
  ├─ Cần tối ưu engagement dài hạn?          │            │
  │     ├─ YES → Reinforcement Learning       │            │
  │     │         (tránh clickbait)           │            │
  │     │                                     │            │
  │     └─ NO → xem tiêu chí tiếp             │            │
  │                                           │            │
  └─ Dataset lớn (>1M interactions)?          │            │
        ├─ YES → Matrix Factorization (ALS)   │            │
        │         + NRMS / NAML ranking        │            │
        │                                     │            │
        └─ NO  → LSTUR (long+short term)      │            │
                  + Content-Based hybrid      ◄────────────┘
```

---

### 10.4 So sánh Chi tiết từng Phương pháp

#### A. Chỉ dùng Content-Based (BERT)

```
Khi nào phù hợp:
  ✓ Ít dữ liệu lịch sử (<10K interactions)
  ✓ Nhiều bài mới mỗi ngày (cold start nặng)
  ✓ Cần giải thích: "Gợi ý vì bạn hay đọc về công nghệ"
  ✓ Hệ thống nhỏ, team nhỏ

Khi nào KHÔNG phù hợp:
  ✗ User có sở thích đa dạng, khó mô tả bằng content
  ✗ Cần khám phá nội dung ngoài vùng sở thích (serendipity)

Latency: thấp (pre-compute embedding, dùng ANN)
Độ phức tạp triển khai: trung bình
```

#### B. Collaborative Filtering (ALS / BPR)

```
Khi nào phù hợp:
  ✓ Có nhiều dữ liệu lịch sử (>100K interactions)
  ✓ User đăng nhập ổn định (có user ID)
  ✓ Muốn khám phá sở thích tiềm ẩn (latent interests)
  ✓ Gợi ý chủ đề/chuyên mục (category-level)

Khi nào KHÔNG phù hợp:
  ✗ Bài mới liên tục (item cold start)
  ✗ User mới chưa có lịch sử
  ✗ Cần real-time cập nhật ngay

Latency: thấp sau khi pre-train (dùng ANN)
Độ phức tạp triển khai: cao (cần retrain định kỳ)
```

#### C. LSTUR (Long + Short Term User Representation)

```
Khi nào phù hợp:
  ✓ User có lịch sử đủ dài (>50 bài đã đọc)
  ✓ Sở thích user thay đổi theo sự kiện (World Cup, bầu cử)
  ✓ Muốn cân bằng: sở thích lâu dài + xu hướng hiện tại
  ✓ Benchmark MIND dataset

Ví dụ:
  Long-term:  user thích [Kinh tế, Thể thao] quanh năm
  Short-term: tháng này đọc nhiều về [Euro 2024]
  → Gợi ý: kinh tế thể thao + tin Euro mới nhất

Latency: trung bình (cần GRU inference)
Độ phức tạp triển khai: cao
```

#### D. NRMS (Multi-Head Self-Attention)

```
Khi nào phù hợp:
  ✓ Cần hiểu mối quan hệ giữa các bài đã đọc
  ✓ Dataset lớn, GPU sẵn có
  ✓ Muốn state-of-the-art trên MIND benchmark
  ✓ Bài báo có tiêu đề/nội dung phong phú

So với LSTUR:
  NRMS:  không phân biệt long/short term → đơn giản hơn
  LSTUR: tường minh tách 2 loại sở thích → interpretable hơn

Latency: cao hơn LSTUR (Transformer nặng hơn GRU)
Độ phức tạp triển khai: cao
```

#### E. Session-Based (SASRec / GRU4Rec)

```
Khi nào phù hợp:
  ✓ User KHÔNG đăng nhập (ẩn danh)
  ✓ Muốn gợi ý theo ngữ cảnh phiên đọc hiện tại
  ✓ Tin tức theo chuỗi sự kiện (breaking news)

Ví dụ:
  Phiên đọc: Động đất → Sóng thần → Cứu hộ → ?
  → SASRec gợi ý: "Thiệt hại được đánh giá", "Viện trợ quốc tế"
  (không cần biết user là ai)

Latency: thấp (chỉ cần phiên hiện tại)
Độ phức tạp triển khai: trung bình
```

#### F. Reinforcement Learning (DQN / LinUCB)

```
Khi nào phù hợp:
  ✓ Muốn tối ưu engagement dài hạn (không chỉ CTR)
  ✓ Có vấn đề clickbait nghiêm trọng
  ✓ Muốn cân bằng explore (khám phá) / exploit (khai thác)
  ✓ Có infrastructure để thu feedback real-time

Khi nào KHÔNG phù hợp:
  ✗ Team nhỏ, ít tài nguyên
  ✗ Cần kết quả nhanh (RL cần nhiều thời gian train)
  ✗ Không có hệ thống thu reward real-time

Latency: cao (cần agent inference)
Độ phức tạp triển khai: rất cao
```

---

### 10.5 Lộ trình Xây dựng theo Giai đoạn

Không cần (và không nên) bắt đầu với mô hình phức tạp nhất:

```
GIAI ĐOẠN 1 — MVP (tuần 1-2)
  └─ Popularity-based + Recency filter
     "Gợi ý 10 bài hot nhất 6 giờ qua"
     → Đơn giản nhất, đủ dùng khi chưa có data

GIAI ĐOẠN 2 — Cá nhân hóa cơ bản (tháng 1)
  └─ Content-Based: BERT embedding tiêu đề
     + Cosine similarity với lịch sử đọc
     → Xử lý được cold start, không cần nhiều data

GIAI ĐOẠN 3 — Cá nhân hóa nâng cao (tháng 2-3)
  └─ LSTUR hoặc NRMS trên MIND-style data
     + Two-tower candidate generation
     → Chính xác hơn, cần >100K interaction logs

GIAI ĐOẠN 4 — Tối ưu hóa (tháng 4+)
  └─ RL-based re-ranker
     + Diversity regularization
     + A/B testing framework
     → Tối ưu engagement dài hạn, chống filter bubble
```

---

### 10.6 Bảng Tổng hợp Chọn Phương pháp

| Tình huống | Phương pháp chính | Phương pháp phụ | Lý do |
|---|---|---|---|
| Khởi đầu, ít data | Popularity + Recency | Content-Based | Không cần lịch sử |
| User ẩn danh | Session-Based (SASRec) | Trending | Chỉ dùng phiên hiện tại |
| Bài mới (<1h) | Content-Based (BERT) | Trending boost | CF không đủ data |
| Bài cũ, user có history | LSTUR / NRMS | CF (ALS) | Tận dụng lịch sử dài hạn |
| Scale lớn (>1M user) | Two-Tower + ANN | NRMS ranking | Tốc độ + chính xác |
| Chống clickbait | RL (DQN/REINFORCE) | Multi-objective loss | Tối ưu reward dài hạn |
| Cần giải thích | Content-Based | Knowledge Graph | Interpretable |
| Đa dạng hóa | Diversity regularization | Causal Inference | Phá filter bubble |

---

### 10.7 Ví dụ Cụ thể: Thiết kế hệ thống cho Báo Điện tử Việt Nam

**Bài toán:** 500K user/ngày, 1000 bài mới/ngày, cần gợi ý real-time

```
Thiết kế đề xuất:

[Candidate Generation - Tầng 1]
  • BM25 recall:       lấy 100 bài theo keyword từ lịch sử đọc
  • Two-Tower ANN:     lấy 200 bài theo embedding similarity
  • Trending filter:   lấy 50 bài hot trong 3 giờ qua
  • Tổng: ~300 bài ứng viên (dedup)

[Ranking - Tầng 2]
  • LSTUR score:       học sở thích dài hạn (thể thao? kinh tế?)
                       + ngắn hạn (tuần này đọc gì?)
  • Recency score:     e^(-0.1 × age_hours)
  • Diversity penalty: -0.2 nếu đã có bài cùng chủ đề trong top-5

  Final score = 0.5 × LSTUR + 0.3 × Recency + 0.2 × Popularity
              - 0.1 × Diversity_penalty

[Output]
  → Top 15 bài hiển thị trên feed

[Cập nhật]
  • LSTUR retrain: 1 lần/ngày (dùng log hôm qua)
  • Two-Tower: retrain 1 lần/tuần
  • BM25 index: cập nhật real-time khi có bài mới
```

> Nguồn: [A Survey of Real-World Recommender Systems — arXiv 2025](https://arxiv.org/html/2509.06002v1) · [Two-Tower Retrieval — Google Cloud](https://docs.cloud.google.com/architecture/implement-two-tower-retrieval-large-scale-candidate-generation) · [Benchmarking News Recommendation in Green AI Era — arXiv 2024](https://arxiv.org/pdf/2403.04736)

---

## Nguồn tham khảo

- [MIND: A Large-scale Dataset for News Recommendation — Microsoft Research](https://www.microsoft.com/en-us/research/publication/mind-a-large-scale-dataset-for-news-recommendation/)
- [NAML: Neural News Recommendation with Attentive Multi-View Learning — arXiv](https://arxiv.org/pdf/2204.04726)
- [LSTUR: Neural News Recommendation with Long- and Short-term User Representations](https://hannlp.github.io/2020-11-11-Some-News-Recommendation-Models/)
- [NRMS: Neural News Recommendation with Multi-Head Self-Attention](https://hannlp.github.io/2020-11-11-Some-News-Recommendation-Models/)
- [UNBERT: User-News Matching BERT for News Recommendation — Semantic Scholar](https://www.semanticscholar.org/paper/UNBERT:-User-News-Matching-BERT-for-News-Zhang-Li/d50f7e9fdcce8bf9f34bac969e091603f4727054)
- [Co-NAML-LSTUR: Combined Model for News Recommendation — arXiv 2025](https://arxiv.org/html/2507.20210)
- [Transformers4NewsRec: A Transformer-based News Recommendation Framework — arXiv 2024](https://arxiv.org/html/2410.13125)
- [A Causal View for Multi-Interest User Modeling in News Recommendation — ACM ICMR 2024](https://dl.acm.org/doi/10.1145/3652583.3658093)
- [Diversification in Session-based News Recommender Systems — arXiv](https://arxiv.org/pdf/2102.03265)
- [Filter bubble effects in news recommendation — Taylor & Francis 2024](https://www.tandfonline.com/doi/full/10.1080/1369118X.2024.2435998)
