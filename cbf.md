# Lọc Dựa Trên Nội Dung: Gợi Ý Dựa Trên Những Gì Bạn Thích

> **Nguồn:** https://www.shaped.ai/blog/content-based-filtering-explained-recommending-based-on-what-you-like  
> **Tác giả:** Tullie Murrell  
> **Ngày:** 8 tháng 7, 2025  
> **Thời gian đọc:** 4 phút

---

Lọc Dựa Trên Nội Dung (Content-Based Filtering - CBF) là một trong những phương pháp cơ bản để xây dựng hệ thống gợi ý. Thay vì dựa vào sở thích của những người dùng tương tự, CBF tập trung vào đặc điểm của các mục mà người dùng đã tương tác để gợi ý các mục khác có thuộc tính tương tự — dù là văn bản, hình ảnh, dữ liệu có cấu trúc hay âm thanh. Bài viết này giới thiệu cách CBF hoạt động, sự phát triển từ kết hợp từ khóa đơn giản đến việc sử dụng các mô hình embedding hiện đại, cùng những thách thức trong quá trình triển khai. Bài cũng trình bày các mẫu thiết kế khác nhau mà Shaped hỗ trợ để áp dụng CBF trong thực tế.

Cá nhân hóa là chìa khóa trong thế giới số ngày nay. Các hệ thống gợi ý đóng vai trò quan trọng, giúp chúng ta khám phá sản phẩm, bài viết, phim ảnh và nhiều nội dung khác. Trong khi Lọc Cộng Tác (Collaborative Filtering) tận dụng "trí tuệ đám đông", thì một phương pháp cơ bản khác lại hoạt động theo cách khác biệt: Lọc Dựa Trên Nội Dung (CBF).

Thay vì xem xét những gì người dùng tương tự thích, CBF tập trung trực tiếp vào bạn và nội dung của những mục bạn đã tương tác tích cực trong quá khứ. Ý tưởng cốt lõi đơn giản nhưng mạnh mẽ: *"Nếu bạn thích cái đó, bạn có thể cũng thích cái này vì chúng có những đặc điểm tương đồng."*

Bài viết này đi sâu vào Lọc Dựa Trên Nội Dung:

- Các nguyên tắc cơ bản và cách thức hoạt động.
- Lịch sử và sự phát triển từ từ khóa đơn giản đến embedding tinh vi.
- Những thách thức khi xây dựng hệ thống CBF từ đầu.
- Cách AI hiện đại (các mô hình Ngôn ngữ và Thị giác) cách mạng hóa việc hiểu nội dung.
- Cách các nền tảng như Shaped triển khai các hướng gợi ý dựa trên nội dung khác nhau.

---

## Lọc Dựa Trên Nội Dung là gì? Ý Tưởng Cốt Lõi

Về bản chất, Lọc Dựa Trên Nội Dung hoạt động dựa trên hai thành phần chính:

**Biểu diễn Mục (Item Representation):** Mỗi mục được mô tả bằng một tập hợp các đặc trưng hoặc thuộc tính. Chúng có thể là:
- **Văn bản:** Từ khóa, mô tả, danh mục, thẻ, thể loại.
- **Có cấu trúc:** Tác giả, đạo diễn, thương hiệu, giá cả, năm sản xuất.
- **Hình ảnh:** Ảnh, hình thu nhỏ của video.
- **Âm thanh:** Đặc trưng âm thanh, siêu dữ liệu thể loại nhạc.

**Hồ Sơ Người Dùng (User Profile):** Một hồ sơ được xây dựng cho mỗi người dùng, tóm tắt sở thích của họ dựa trên các đặc trưng của những mục họ đã thích hoặc tương tác tích cực trước đó.

Quá trình gợi ý thường theo các bước sau:

1. **Phân tích Nội dung:** Trích xuất các đặc trưng liên quan từ những mục người dùng đã thích.
2. **Xây dựng Hồ sơ:** Tạo hoặc cập nhật hồ sơ người dùng dựa trên các đặc trưng này (ví dụ: vector đặc trưng có trọng số).
3. **Khớp Nội dung:** So sánh hồ sơ người dùng với biểu diễn đặc trưng của các mục khác mà người dùng chưa xem.
4. **Gợi ý:** Đề xuất các mục có đặc trưng phù hợp chặt chẽ nhất với hồ sơ người dùng, thường sử dụng một độ đo tương đồng.

---

## Hành Trình của Lọc Dựa Trên Nội Dung: Từ Từ Khóa đến Ngữ Nghĩa

CBF đã tồn tại từ những ngày đầu của truy xuất thông tin và hệ thống gợi ý.

### Thuở Ban Đầu (Từ khóa & TF-IDF)

Các dạng sớm nhất phụ thuộc nhiều vào đặc trưng văn bản. Kỹ thuật TF-IDF (Tần suất Từ - Tần suất Nghịch Đảo Tài liệu) được dùng để biểu diễn các mục (bài báo, tài liệu) dưới dạng vector trọng số từ khóa. Hồ sơ người dùng cũng là các vector tóm tắt các từ khóa quan trọng từ những mục đã thích. Độ tương đồng thường được tính bằng Cosine Similarity.

**Thách thức:** Phương pháp này gặp khó với từ đồng nghĩa (ví dụ: "phim" và "movie"), đa nghĩa (từ có nhiều nghĩa), và khó hiểu các mối quan hệ ngữ nghĩa sâu hơn. Nó cũng không dễ xử lý các đặc trưng phi văn bản.

### Mô Hình Không Gian Vector & Kỹ Thuật Đặc Trưng

Khái niệm mở rộng để tích hợp thêm các đặc trưng có cấu trúc (thể loại, diễn viên, thương hiệu). Điều này đòi hỏi kỹ thuật đặc trưng đáng kể — xác định thủ công cách biểu diễn các loại nội dung khác nhau và kết hợp chúng thành một biểu diễn mục thống nhất.

**Thách thức:** Kỹ thuật đặc trưng tốn nhiều công sức, phụ thuộc vào từng lĩnh vực và dễ bị lỗi. Kết hợp các đặc trưng không đồng nhất (văn bản, danh mục, giá trị số) thành điểm tương đồng có ý nghĩa là rất phức tạp.

### Nhu Cầu Hiểu Biết Sâu Hơn

Khi nội dung trở nên phong phú hơn (hình ảnh, mô tả phức tạp) và kỳ vọng của người dùng tăng cao, những hạn chế của khớp đặc trưng đơn giản trở nên rõ ràng. Cần có các mô hình có thể hiểu *ý nghĩa* đằng sau nội dung, không chỉ là từ khóa bề mặt.

---

## Cách Lọc Dựa Trên Nội Dung Hoạt Động: Các Bước & Thành Phần Chính

### 1. Biểu Diễn Mục / Trích Xuất Đặc Trưng

Đây là bước quan trọng nhất. Bạn cần chuyển đổi nội dung thô của mục thành định dạng có cấu trúc phù hợp để so sánh.

- **Văn bản:** Làm sạch văn bản, tokenize, loại bỏ stop words, áp dụng TF-IDF hoặc dùng kỹ thuật nâng cao như Bag-of-Words.
- **Phân loại:** Dùng one-hot encoding hoặc phổ biến hơn là học embeddings cho danh mục/thẻ.
- **Số học:** Chuẩn hóa giá trị.
- **Đầu ra:** Thường là vector hồ sơ mục `v_i` cho mỗi mục `i`.

### 2. Xây Dựng Hồ Sơ Người Dùng

Tổng hợp các vector đặc trưng của những mục mà người dùng `u` đã tương tác tích cực.

- **Cách đơn giản:** Lấy trung bình các vector mục `v_i` mà người dùng đã thích.
- **Cách có trọng số:** Cho trọng số lớn hơn với mục được đánh giá cao hoặc tương tác gần đây hơn.
- **Đầu ra:** Vector hồ sơ người dùng `p_u`.

### 3. Tính Độ Tương Đồng

Đo độ tương đồng giữa vector hồ sơ người dùng `p_u` và vector `v_j` của mỗi mục ứng viên `j`.

- **Độ đo phổ biến:** Cosine Similarity: `similarity(p_u, v_j) = (p_u · v_j) / (||p_u|| ||v_j||)`
  - Đo góc giữa các vector, nắm bắt hướng thay vì độ lớn.
- Các độ đo khác như **Dot Product** hoặc **Khoảng cách Euclidean** cũng có thể được dùng tùy thuộc vào biểu diễn vector.

### 4. Tạo Gợi Ý

Xếp hạng các mục ứng viên `j` dựa trên điểm tương đồng với hồ sơ người dùng `p_u`. Trình bày top-N mục tương đồng nhất dưới dạng gợi ý.

---

## Sự Trỗi Dậy của Deep Learning: Hiểu Nội Dung Tốt Hơn

Deep learning đã cải thiện đáng kể CBF bằng cách cung cấp nhiều cách biểu diễn nội dung phong phú hơn:

- **Embeddings là chủ đạo:** Thay vì vector TF-IDF thưa thớt hoặc đặc trưng được kỹ thuật thủ công, các mô hình deep learning học được các embedding dày đặc — vector chiều thấp nắm bắt ý nghĩa ngữ nghĩa. Các mục có ý nghĩa tương tự (dù dùng từ khác nhau) sẽ có embedding gần nhau trong không gian vector.

- **Mô hình Ngôn ngữ (LLMs):** Các mô hình như Word2Vec, GloVe và đặc biệt là các Transformer được pre-train (BERT, Sentence-BERT, RoBERTa...) có thể xử lý tiêu đề, mô tả, đánh giá và thẻ của mục để tạo ra embedding văn bản mạnh mẽ. Chúng hiểu ngữ cảnh, từ đồng nghĩa và các sắc thái tốt hơn nhiều so với các phương pháp cũ.

- **Mô hình Thị giác:** CNN (ResNet, EfficientNet) và Vision Transformer (ViT) có thể xử lý hình ảnh mục để tạo visual embedding. Điều này cho phép gợi ý các sản phẩm tương tự về mặt thị giác (ví dụ: thời trang, nội thất).

- **Mô hình Đa phương thức:** Các mô hình như CLIP học joint embedding cho cả hình ảnh và văn bản, cho phép gợi ý dựa trên độ tương đồng chéo phương thức (ví dụ: tìm sản phẩm khớp với mô tả văn bản hoặc ngược lại).

Các embedding nâng cao này có thể được sử dụng trực tiếp làm biểu diễn mục (`v_i`) trong pipeline CBF, dẫn đến các gợi ý chính xác và tinh tế hơn.

---

## Xây Dựng Lọc Dựa Trên Nội Dung Từ Đầu: Những Thách Thức

Dù khái niệm đơn giản, xây dựng hệ thống CBF mạnh mẽ không hề dễ dàng:

- **Kỹ thuật/Trích xuất Đặc trưng:** Vẫn là thách thức ngay cả với deep learning. Chọn đúng mô hình pre-trained, tinh chỉnh chúng và quyết định dùng đặc trưng nào (văn bản, hình ảnh, có cấu trúc) đòi hỏi chuyên môn cao.
- **Khả năng Mở rộng:** Tính độ tương đồng giữa hồ sơ người dùng và hàng triệu vector mục theo thời gian thực rất tốn kém về mặt tính toán. Cần các kỹ thuật như tìm kiếm Nearest Neighbor xấp xỉ (ANN) trên embedding mục.
- **Động lực Hồ sơ Người dùng:** Hồ sơ nên cập nhật nhanh đến đâu với sở thích mới? Trọng số nào cho tương tác cũ so với mới? Giữ hồ sơ luôn mới và liên quan là thách thức phức tạp.
- **Quá Chuyên biệt (Filter Bubble):** CBF có xu hướng gợi ý những mục rất tương đồng với tương tác trong quá khứ. Điều này hạn chế khám phá và sự bất ngờ, nhốt người dùng trong phạm vi nội dung hẹp.
- **Phụ thuộc Chất lượng Nội dung:** Chất lượng gợi ý phụ thuộc nhiều vào chất lượng và độ phong phú của nội dung/siêu dữ liệu mục. Mô tả nghèo nàn dẫn đến gợi ý kém.
- **Cold Start Người dùng:** Dù CBF xử lý tốt cold start mục (miễn là mục mới có đặc trưng nội dung), nó vẫn cần một số lịch sử tương tác của người dùng để xây dựng hồ sơ ban đầu.

---

## Lọc Dựa Trên Nội Dung Trong Thực Tế: Cách Tiếp Cận Của Shaped

Các nền tảng hiện đại như Shaped trừu tượng hóa nhiều phức tạp triển khai và cung cấp các cách linh hoạt để tận dụng độ tương đồng nội dung. Shaped hỗ trợ nhiều `policy_types` trong cấu hình `embedding_policy` và `scoring_policy` để triển khai các hướng logic dựa trên nội dung khác nhau:

### `item-content-similarity`

**Cách hoạt động:** Policy này theo sát mẫu CBF truyền thống nhưng dùng embedding hiện đại. Nó tính embedding mục dựa trên thuộc tính mục (ví dụ: text embedding từ mô tả, categorical embedding từ thẻ). Embedding người dùng (hồ sơ) sau đó được tính bằng cách pooling các embedding của những mục người dùng đã tương tác.

**Trường hợp sử dụng:** CBF cổ điển — gợi ý những mục tương đồng với những gì người dùng đã tương tác trước đó.

### `user-content-similarity`

**Cách hoạt động:** Policy này đảo chiều góc nhìn. Nó tính embedding người dùng dựa trên thuộc tính người dùng. Embedding mục sau đó được suy ra bằng cách pooling embedding của những người dùng đã tương tác với mục đó.

**Trường hợp sử dụng:** Hữu ích khi thuộc tính người dùng phong phú và bạn muốn tìm những mục được thích bởi người dùng có thuộc tính tương đồng.

```yaml
model:
  name: user-content-recs
  policy_configs:
    scoring_policy:
      policy_type: user-content-similarity
      pool_fn: mean
      distance_fn: cosine
```

### `user-item-content-similarity`

**Cách hoạt động:** Policy này thực hiện so sánh trực tiếp giữa thuộc tính người dùng và thuộc tính mục. Nó tính embedding người dùng từ đặc trưng người dùng và embedding mục từ đặc trưng mục một cách độc lập. Độ tương đồng sau đó được tính trực tiếp giữa hai embedding này.

**Trường hợp sử dụng:** Mạnh mẽ khi thuộc tính người dùng và mục tồn tại trong ngữ cảnh được căn chỉnh (ví dụ: thuộc tính 'sở thích' của người dùng so với thuộc tính 'thẻ' của mục). Không cần dựa vào lịch sử tương tác trong quá khứ cho việc tính toán độ tương đồng, chỉ dựa vào thuộc tính vốn có.

```yaml
model:
  name: direct-content-match
  policy_configs:
    scoring_policy:
      policy_type: user-item-content-similarity
      distance_fn: cosine
```

Các policy này cho phép bạn tận dụng nội dung theo những cách tinh vi, thường sử dụng các mô hình embedding mạnh mẽ được pre-train hoặc fine-tune do nền tảng quản lý, mà không cần tự xây dựng pipeline trích xuất đặc trưng và tính toán độ tương đồng.

---

## Ưu điểm và Nhược điểm của Lọc Dựa Trên Nội Dung

### Ưu điểm

- ✅ **Xử lý tốt Cold Start Mục:** Có thể gợi ý mục mới ngay lập tức nếu chúng có đặc trưng nội dung.
- ✅ **Độc lập với Người dùng khác:** Gợi ý cho một người dùng không phụ thuộc vào dữ liệu của người dùng khác.
- ✅ **Khả năng Giải thích:** Gợi ý thường có thể được giải thích dựa trên đặc trưng mục (ví dụ: "Gợi ý vì bạn thích các mục thuộc thể loại X").
- ✅ **Không Thiên vị Phổ biến:** Không ưu tiên mục phổ biến một cách tự nhiên; gợi ý chỉ dựa trên độ tương đồng đặc trưng.

### Nhược điểm

- ❌ **Kỹ thuật/Biểu diễn Đặc trưng:** Chất lượng phụ thuộc nhiều vào đặc trưng nội dung có sẵn và phương pháp biểu diễn chúng.
- ❌ **Quá Chuyên biệt:** Có thể dẫn đến gợi ý hẹp và hạn chế khám phá ("filter bubble").
- ❌ **Cold Start Người dùng:** Vẫn cần một số tương tác ban đầu của người dùng để xây dựng hồ sơ có ý nghĩa cho các policy như `item-content-similarity`.
- ❌ **Không Tận dụng Thông tin Cộng tác:** Bỏ lỡ việc dự đoán sở thích dựa trên những gì người dùng tương tự thích, điều mà thường nắm bắt được các sở thích tinh tế không rõ ràng từ nội dung đơn thuần.

---

## Kết Luận: Một Công Cụ Quan Trọng trong Bộ Công Cụ Gợi Ý

Lọc Dựa Trên Nội Dung là kỹ thuật gợi ý nền tảng tận dụng đặc điểm của mục để dự đoán sở thích người dùng. Từ nguồn gốc khớp từ khóa đơn giản đến các triển khai hiện đại sử dụng embedding deep learning tinh vi cho văn bản và hình ảnh, nó cung cấp một cách mạnh mẽ để cá nhân hóa trải nghiệm dựa trên hồ sơ khẩu vị cá nhân được suy ra từ nội dung.

Dù đối mặt với những thách thức như quá chuyên biệt và yêu cầu dữ liệu nội dung chất lượng tốt, khả năng xử lý mục mới và cung cấp gợi ý có thể giải thích khiến nó trở nên vô giá. Các nền tảng như Shaped đơn giản hóa việc triển khai, cung cấp nhiều chiến lược tương đồng nội dung phù hợp với các trường hợp sử dụng khác nhau. Thông thường, các hệ thống gợi ý mạnh mẽ nhất kết hợp cả hai phương pháp — CBF và Lọc Cộng Tác cùng các kỹ thuật khác để mang lại trải nghiệm người dùng liên quan và hấp dẫn nhất.

---

*Sẵn sàng xây dựng gợi ý thông minh hơn dựa trên nội dung người dùng yêu thích? [Yêu cầu demo Shaped](https://www.shaped.ai) ngay hôm nay.*
