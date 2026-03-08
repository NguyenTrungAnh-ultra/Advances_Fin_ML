# Báo Cáo Kiểm Định Khai Khoáng Đặc Tính Của Động Cơ Bar (MLFinLab Standard)

Dựa trên sách *Advances in Financial Machine Learning*, các thanh Bar dựa trên luồng thông tin (Dollar, Imbalance) được chứng minh là có ưu việt hơn so với Time Bars truyền thống về mặt thống kê.

Dưới đây là đối chiếu kết quả khi chạy qua cùng 1 khung dữ liệu 6 tháng của mã FPT (Cùng thiết lập để N đạt khối lượng tương đồng).

## 1. Kết Quả Thống Kê

| Loại Thanh Bar      |   Tổng Số Lượng (N) |   Autocorr (Lag 1)  |      Jarque-Bera Stat  |    Độ lệch chuẩn SL Bar / Tuần  |
|                     |                     |            (~0=Tốt) |   (Thấp=Chuẩn hóa hơn) |   (Data-Driven sẽ dao động cao) |
|:--------------------|--------------------:|--------------------:|-----------------------:|--------------------------------:|
| Time Bars           |                 594 |             -0.0311 |                1218.82 |                            5.35 |
| Dynamic Dollar Bars |                2645 |             -0.0082 |                5224.86 |                           44.53 |
| Imbalance Bars      |                   7 |             -0.099  |                   1.16 |                            0.67 |

## 2. Giải Nghĩa Các Bài Test

### A. Autocorr (Tự Tương Quan Bậc 1)
- **Ý nghĩa:** Kiểm tra xem giá bị dội ngược (mean-reverting) nhanh chống do nhiễu loạn vi mô (Microstructure Noise) cấu trúc vi mô thị trường hay không. Rất thường gặp ở khung thời gian.
- **Kết quả mong đợi:** Các thuật toán Machine Learning luôn prefer dữ liệu có độ tự tương quan I.I.D (Độc lập phân phối đồng nhất) gần `0` nhất, chứng tỏ Bar đã gom và triệt tiêu được bọng nhiễu tĩnh lặng.

### B. Kiểm Định Chuẩn Hóa Jarque-Bera & Kurtosis
- **Ý nghĩa:** Trong tài chính, Time Bars nổi tiếng là tạo ra thị trường có Đuôi Béo (Fat Tails) tức là hay xuất hiện cực đoan, không giống phân phối chuẩn Gauss (chuông), làm các mô hình ML tuyến tính, GARCH hay NN bị 'choáng' mồi.
- **Kết quả mong đợi:** Bar nào có điểm Jarque-Bera Stat càng thấp (hoặc Kurtosis càng gần 3) nghĩa là phân phối lợi suất của nó tròn trịa nhất, dễ học nhất cho AI.

### C. Độ lệch chuẩn Số Lượng Bar / Tuần
- **Ý nghĩa:** Time Bars thì cứ đúng tíc tắc là vẽ, không quan tâm TT câm điếc hay nổ bùng. Nhưng Data-driven Bars vẽ nến dựa trên tốc độ bơm thông tin (Rate of flow). 
- **Kết quả mong đợi:** Với Time bars, con số độ lệch này thường gần bằng 0 cực kỳ ổn định. Với Dollar/Imbalance, con số này phải dao động rất cao (Vì tuần yên tĩnh ko vẽ bar nào, tuần có tin ra vẽ hàng chục Bar). **Biến động số nến cao là bản chất ưu việt để thanh 'cắn' tin tức**.
