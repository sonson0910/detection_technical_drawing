**TECHNICAL ASSESSMENT**

**Object Detection & OCR System for Engineering Drawings**

Bài kiểm tra kỹ thuật dành cho ứng viên vị trí Computer Vision / AI
Engineer

  **Thời gian làm bài**    **96 giờ kể từ khi nhận đề**
  ------------------------ -----------------------------------------
  **Loại bài**             Take-home Project
  **Ngôn ngữ lập trình**   Python (ưu tiên), hoặc ngôn ngữ phù hợp
  **Nộp bài**              GitHub repo + URL web demo

## **1. Tổng quan bài toán**

Bạn cần xây dựng một hệ thống tự động phát hiện và cắt rời các đối tượng
trong bản vẽ kỹ thuật (engineering drawing). Hệ thống cần nhận diện,
định vị và trích xuất 3 loại đối tượng:

-   PartDrawing --- vùng chứa bản vẽ chi tiết kỹ thuật

-   Note --- vùng ghi chú, chú thích

-   Table --- vùng bảng dữ liệu kỹ thuật

> ![](media/image1.png){width="6.5in" height="3.6527777777777777in"}

Ngoài việc detect và crop, hệ thống cần thực hiện OCR để trích xuất nội
dung văn bản từ các vùng Note và Table, đặc biệt phải giữ nguyên cấu
trúc bảng (rows, columns, cell alignment).

+----------------------------------------------------------------------+
| **Ràng buộc quan trọng về framework:**                               |
|                                                                      |
| Không sử dụng YOLO --- YOLO không cho phép thương mại hóa nếu không  |
| được cấp phép hoặc sử dụng public dataset.                           |
|                                                                      |
| Khuyến nghị: Detectron2. Cũng có thể dùng RT-DETR, DINO, MMDetection |
| với backbone phù hợp,                                                |
|                                                                      |
| hoặc bất kỳ framework detection nào có giấy phép thương mại tương    |
| thích.                                                               |
+----------------------------------------------------------------------+

## **2. Tài nguyên được cung cấp**

**Dataset (ảnh/nhãn):**
[[https://drive.google.com/drive/folders/1hyl_NsBOvVcyWqDLeSs32d32p6vY9ggQ?usp=sharing]{.ul}](https://drive.google.com/drive/folders/1hyl_NsBOvVcyWqDLeSs32d32p6vY9ggQ?usp=sharing)

**Expected output mẫu:**
[[https://drive.google.com/drive/folders/1un6okShWQeQkS6YA43mCz5QMTypB1mUS?usp=drive_link]{.ul}](https://drive.google.com/drive/folders/1un6okShWQeQkS6YA43mCz5QMTypB1mUS?usp=drive_link)

+----------------------------------------------------------------------+
| Hãy xem kỹ thư mục Expected Output trước khi bắt đầu để hiểu đúng    |
| định dạng                                                            |
|                                                                      |
| và chất lượng kỳ vọng của hệ thống --- bao gồm cấu trúc file crop,   |
| JSON output và OCR output.                                           |
+----------------------------------------------------------------------+

## **3. Yêu cầu kỹ thuật chi tiết**

**3.1 Object Detection Pipeline**

1.  Train model nhận diện 3 class: PartDrawing, Note, Table trên dataset
    > được cung cấp.

2.  Đầu ra cho mỗi ảnh: danh sách bounding box kèm class label và
    > confidence score.

3.  Tự động crop từng đối tượng detected thành ảnh con riêng biệt.

4.  Xuất JSON chứa thông tin tọa độ, class, confidence của từng đối
    > tượng.

**3.2 OCR Pipeline**

5.  Thực hiện OCR trên tất cả vùng được detect là Note và Table.

6.  Với vùng Table: kết quả OCR phải giữ nguyên cấu trúc bảng (rows,
    > columns).

7.  Kết quả OCR được tích hợp vào JSON output chung của hệ thống.

8.  Ưu tiên độ chính xác tối đa --- có thể kết hợp nhiều OCR engine hoặc
    > post-processing.

**3.3 JSON Output format**

Mỗi ảnh đầu vào cần tạo ra 1 file JSON với cấu trúc tối thiểu như sau:

{ \"image\": \"drawing_001.jpg\", \"objects\": \[ { \"id\": 1,
\"class\": \"Table\", \"confidence\": 0.97, \"bbox\": { \"x1\": 120,
\"y1\": 340, \"x2\": 680, \"y2\": 520 }, \"ocr_content\": \"\...\" },
\... \]}

## **4. Yêu cầu Web Demo**

Sau khi hoàn thành model, bạn cần deploy một web demo nhỏ lên Vercel,
Render, HuggingFace Spaces hoặc bất kỳ nền tảng miễn phí nào. Giao diện
cần có đủ các thành phần sau:

-   Upload zone: ô cho phép upload ảnh bản vẽ và nút Gửi / Detect để
    > trigger pipeline.

-   Visualization: hiển thị ảnh kết quả với bounding box đã vẽ sẵn, phân
    > biệt màu theo class.

-   JSON panel: hiển thị JSON tọa độ và thông tin của từng đối tượng
    > được detect.

-   OCR panel: hiển thị nội dung OCR của từng vùng Note và Table được
    > phát hiện.

## **5. Tiêu chí chấm điểm**

Bài làm được đánh giá trên tập test bí mật do người phỏng vấn chuẩn bị.
Tập test sẽ có phân phối dữ liệu tương đồng với dataset cung cấp --- hãy
đảm bảo model hoạt động tốt và ổn định trên phân phối này.

  **Tiêu chí**                        **Mô tả**                                     **Mức độ**
  ----------------------------------- --------------------------------------------- -----------------
  **Detection quality (IoU / mAP)**   Bounding box chính xác trên test set bí mật   Quan trọng nhất
  **OCR accuracy**                    Độ chính xác nội dung Note và Table           Quan trọng nhất
  **Table structure preservation**    Giữ nguyên cấu trúc hàng/cột sau OCR          Quan trọng
  **Web demo functionality**          Upload, detect, hiển thị đúng yêu cầu         Bắt buộc
  **Code quality & methodology**      Approach, documentation, phương pháp bổ trợ   Bonus

+----------------------------------------------------------------------+
| Bài làm càng tốt --- offer càng tốt.                                 |
|                                                                      |
| Hãy ưu tiên chất lượng detection và OCR. Các phương pháp bổ trợ sáng |
| tạo                                                                  |
|                                                                      |
| (data augmentation, ensemble, post-processing, confidence            |
| calibration\...) sẽ được đánh giá cao.                               |
+----------------------------------------------------------------------+

## **6. Yêu cầu nộp bài**

9.  GitHub repository chứa toàn bộ source code, có thể chạy được.

10. File README.md hướng dẫn cài đặt môi trường, train model và chạy
    > inference pipeline.

11. Link model weights (Google Drive, HuggingFace Hub\...) hoặc script
    > tải về tự động.

12. URL web demo đã được deploy, hoạt động ổn định.

13. Báo cáo ngắn (trong README hoặc file riêng) mô tả: approach, các thử
    > nghiệm, kết quả đạt được và hướng cải thiện.

+----------------------------------------------------------------------+
| **Thông báo hoàn thành bài làm**                                     |
|                                                                      |
| Sau khi hoàn thiện toàn bộ bài làm, vui lòng gửi email thông báo kèm |
| link GitHub và link web demo tới:                                    |
|                                                                      |
| **Email:                                                             |
| [[duy.hoang\@sotatek.com]{.ul}](mailto:duy.hoang@sotatek.com)**      |
|                                                                      |
| Email cần bao gồm: link GitHub repo, link web demo, và bất kỳ ghi    |
| chú nào bạn muốn chia sẻ thêm về bài làm.                            |
+----------------------------------------------------------------------+

**Chúc bạn may mắn!**

Hy vọng chúng ta có cơ hội làm việc cùng nhau.
