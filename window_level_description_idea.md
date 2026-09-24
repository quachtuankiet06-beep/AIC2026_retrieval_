# Ý tưởng Window-Level Description cho Video Retrieval và Reranking

## 1. Mục tiêu

Ý tưởng này nhằm cải thiện tầng rerank trong pipeline truy hồi video bằng cách thay vì mô tả riêng từng keyframe, hệ thống sẽ **gom các keyframe gần nhau thành từng cửa sổ ngữ nghĩa (window)** và sinh mô tả chung cho cả cửa sổ đó.

Mục tiêu chính:

- Tăng tính ngữ nghĩa khi các keyframe chỉ thay đổi nhẹ theo thời gian.
- Giảm chi phí captioning / description.
- Giảm nhiễu ở mức keyframe đơn lẻ.
- Tạo thêm một lớp tín hiệu semantic để hỗ trợ reranking.

## 2. Ý tưởng cốt lõi

Mỗi video thường chứa nhiều keyframe. Nhiều keyframe liên tiếp có thể gần như cùng một cảnh, chỉ khác chút về góc nhìn, vị trí, hoặc mức chuyển động.

Thay vì sinh description cho từng keyframe, ta sẽ:

1. Trích embedding của keyframe bằng các model sẵn có như **DFN5B** và **SigLIP2**.
2. Gom các keyframe có embedding tương đồng vào cùng một **window**.
3. Dùng một model local như **PaliGemma-224** để sinh mô tả tiếng Anh cho cả window.
4. Gán cùng một description cho toàn bộ keyframe trong window đó.
5. Dùng description này để so semantic với query trong tầng rerank.

## 3. Tại sao cách này hợp lý

### 3.1. Tăng tính ổn định ngữ nghĩa
Một keyframe đơn lẻ đôi khi bị lệch do:
- motion blur,
- thay đổi nhỏ về khung hình,
- góc quay khác nhẹ,
- vật thể bị che một phần.

Khi gom theo cửa sổ, description sẽ phản ánh **cảnh chung**, ổn định hơn so với từng frame riêng lẻ.

### 3.2. Giảm chi phí
Nếu một video có hàng chục keyframe gần như trùng cảnh, việc caption từng frame là rất tốn kém. Window-level description giúp giảm số lần gọi model sinh mô tả.

### 3.3. Hợp với reranking
Sau khi có description cho cửa sổ, ta có thêm một feature semantic mạnh để:
- so sánh query với window,
- xếp hạng candidate tốt hơn,
- hỗ trợ các trường hợp object / OCR / metadata chưa đủ rõ.

## 4. Vai trò của DFN5B và SigLIP2

Hai model này không dùng để sinh description trực tiếp, mà dùng để tạo embedding cho keyframe.

Ý tưởng là:
- Nếu hai keyframe có embedding gần nhau ở cả DFN5B và SigLIP2,
- và chúng cũng gần nhau theo trục thời gian,
- thì có thể coi chúng thuộc cùng một scene/window.

Điều này giúp việc gom nhóm không phụ thuộc hoàn toàn vào caption model.

## 5. Vai trò của PaliGemma-224

PaliGemma-224 sẽ được dùng để sinh mô tả tiếng Anh cho mỗi window.

Lý do chọn hướng này:
- chạy local được,
- chi phí thấp hơn caption từng keyframe,
- phù hợp với input dạng một hoặc vài frame đại diện của window,
- output tiếng Anh dễ dùng cho semantic matching và downstream reranking.

### Đặc điểm mong muốn của description
Description nên:
- ngắn gọn,
- tập trung vào người, vật, hành động, bối cảnh,
- giữ tính quan sát trực tiếp,
- viết bằng tiếng Anh thống nhất.

Ví dụ output mong muốn:

> Two women are feeding goats in a spacious farm with rows of pens and a covered roof.

## 6. Pipeline đề xuất

### Bước 1: Trích embedding keyframe
Với mỗi keyframe:
- lấy embedding từ DFN5B,
- lấy embedding từ SigLIP2.

### Bước 2: Gom nhóm thành window
Dựa trên:
- cosine similarity của embedding,
- khoảng cách thời gian giữa các keyframe,
- có thể thêm ràng buộc không gộp quá rộng.

### Bước 3: Sinh description cho window
Chọn:
- một frame đại diện, hoặc
- vài frame đại diện của window,

rồi đưa vào PaliGemma-224 để sinh description tiếng Anh cho cả cửa sổ.

### Bước 4: Gán description cho các keyframe trong window
Mỗi keyframe trong cùng window sẽ:
- cùng `window_id`,
- cùng `window_description`,
- cùng `window_embedding` nếu cần.

### Bước 5: Rerank bằng semantic description
So sánh query với description của window/candidate:
- query embedding,
- description embedding,
- hoặc kết hợp với object / metadata / OCR / ASR.

## 7. Cách dùng trong reranking

Trong reranking, thay vì chỉ dựa vào:
- CLIP score,
- object score,
- metadata score,
- OCR score,
- ASR score,

ta có thể thêm:

- `window_description_score`

hoặc

- `semantic_window_score`

Điểm này đo mức khớp giữa query và mô tả cửa sổ.

Một công thức đơn giản:

```text
final_score = multimodal_score + λ * window_description_score
```

Trong đó `λ` nên nhỏ hơn các tín hiệu chính để không làm lệch ranking quá mạnh.

## 8. Lợi ích kỳ vọng

- Giảm lặp keyframe trong cùng một cảnh.
- Tăng khả năng khớp ngữ nghĩa với query.
- Giảm nhiễu khi keyframe riêng lẻ không đủ đại diện.
- Giảm chi phí captioning so với mô tả từng keyframe.
- Tạo thêm một lớp semantic trung gian giữa retrieval và reranking.

## 9. Rủi ro và hạn chế

### 9.1. Window quá lớn
Nếu cửa sổ quá rộng, description có thể quá chung chung và mất chi tiết quan trọng.

### 9.2. Window quá nhỏ
Nếu cửa sổ quá nhỏ, lợi ích giảm và gần giống caption từng keyframe.

### 9.3. Mô tả chung bỏ sót chi tiết
Một window có thể chứa một keyframe rất quan trọng nhưng description chung lại không nhấn mạnh đúng chi tiết đó.

### 9.4. Chất lượng caption phụ thuộc model local
Nếu PaliGemma sinh caption không ổn ở một số trường hợp, description window sẽ kéo theo rerank sai.

## 10. Hướng triển khai an toàn

Nên triển khai theo từng bước nhỏ:

### Giai đoạn 1
Chỉ gom window và sinh description cho top candidate.

### Giai đoạn 2
Dùng description window để rerank top-N candidate.

### Giai đoạn 3
Mở rộng ra toàn bộ pipeline nếu thấy hiệu quả ổn định.

## 11. Schema gợi ý

Mỗi window có thể lưu:

```json
{
  "window_id": "L27_V014_win03",
  "video_id": "L27_V014",
  "keyframe_ids": [316, 317, 318],
  "representative_keyframes": ["..."],
  "window_description": "Two women are feeding goats in a farm.",
  "window_embedding_source": ["dfn5b", "siglip2"],
  "window_score": 0.87
}
```

## 12. Kết luận

Ý tưởng window-level description là một hướng hợp lý để nâng chất lượng rerank.

Nó giữ được:
- độ ổn định ngữ nghĩa,
- chi phí thấp hơn caption từng keyframe,
- khả năng kết hợp tốt với các tín hiệu retrieval hiện tại.

Nếu triển khai cẩn thận, đây có thể trở thành một tầng semantic mạnh hơn so với việc cộng bonus thủ công ở rerank.
