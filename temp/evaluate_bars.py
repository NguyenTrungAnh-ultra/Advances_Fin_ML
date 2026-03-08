import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import sys
from datetime import timedelta

# Thêm path
project_root = r"E:\Projects\adv_ml_fin"
if project_root not in sys.path:
    sys.path.append(project_root)

# Import module
from src.models.preprocess.info_driven import TimeBar, DollarBar

# Tạo thư mục temp
out_dir = r"E:\Projects\adv_ml_fin\temp"
os.makedirs(out_dir, exist_ok=True)

print("Đang tải dữ liệu...")
fpt_path = os.path.join(project_root, 'datasets', 'stocks', 'FPT.csv')
df = pd.read_csv(fpt_path)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
df.sort_index(inplace=True)

df['typical_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4.0
df['dollar_value'] = df['typical_price'] * df['volume']

# Lấy 6 tháng gần nhất để có đủ độ dài kiểm định statistical
df = df[df.index.date >= df.index.date.max() - timedelta(days=180)]

print("1. Đang sinh Time Bars...")
tb = TimeBar.time_bar(df, expected_bars=3078)

print("2. Đang sinh Dynamic Dollar Bars...")
ddb = DollarBar.dynamic_dollar_bars(df, rolling_window=20, n_target=25)

print("3. Đang sinh Imbalance Bars...")
# Giảm initial_T_guess từ 100 xuống 20, span từ 100 xuống 20 để bar nhạy hơn với nến 1p
ib = DollarBar.imbalance(df, initial_T_guess=20, span=20)

brands = {'Time Bars': tb, 'Dynamic Dollar Bars': ddb, 'Imbalance Bars': ib}

# Tính toán các giá trị thống kê
results = []
for name, bars in brands.items():
    if bars.empty:
        continue
    ret = np.log(bars['close'] / bars['close'].shift(1)).dropna()
    
    # 1. Normality (Jarque-Bera)
    jb_stat, jb_p = stats.jarque_bera(ret)
    
    # 2. Autocorrelation (lag 1)
    autocorr = ret.autocorr(lag=1)
    
    # 3. Bar Count Std (Weekly)
    # Đo lường sự ổn định về 'tần suất nạp thông tin'
    counts = bars.resample('W').size()
    count_std = counts.std()
    
    results.append({
        'Loại Thanh Bar': name,
        'Tổng Số Lượng (N)': len(bars),
        'Autocorr (Lag 1) \n(~0=Tốt)': f"{autocorr:.4f}",
        'Jarque-Bera Stat \n(Thấp=Chuẩn hóa hơn)': f"{jb_stat:.2f}",
        'Độ lệch chuẩn SL Bar / Tuần \n(Data-Driven sẽ dao động cao)': f"{count_std:.2f}"
    })

# ======== Xuất Báo Cáo ========
md_path = os.path.join(out_dir, "bar_analysis_report.md")
with open(md_path, 'w', encoding='utf-8') as f:
    f.write("# Báo Cáo Kiểm Định Khai Khoáng Đặc Tính Của Động Cơ Bar (MLFinLab Standard)\n\n")
    f.write("Dựa trên sách *Advances in Financial Machine Learning*, các thanh Bar dựa trên luồng thông tin (Dollar, Imbalance) được chứng minh là có ưu việt hơn so với Time Bars truyền thống về mặt thống kê.\n\n")
    f.write("Dưới đây là đối chiếu kết quả khi chạy qua cùng 1 khung dữ liệu 6 tháng của mã FPT (Cùng thiết lập để N đạt khối lượng tương đồng).\n\n")
    
    f.write("## 1. Kết Quả Thống Kê\n\n")
    df_res = pd.DataFrame(results)
    f.write(df_res.to_markdown(index=False))
    f.write("\n\n")
    
    f.write("## 2. Giải Nghĩa Các Bài Test\n\n")
    f.write("### A. Autocorr (Tự Tương Quan Bậc 1)\n")
    f.write("- **Ý nghĩa:** Kiểm tra xem giá bị dội ngược (mean-reverting) nhanh chống do nhiễu loạn vi mô (Microstructure Noise) cấu trúc vi mô thị trường hay không. Rất thường gặp ở khung thời gian.\n")
    f.write("- **Kết quả mong đợi:** Các thuật toán Machine Learning luôn prefer dữ liệu có độ tự tương quan I.I.D (Độc lập phân phối đồng nhất) gần `0` nhất, chứng tỏ Bar đã gom và triệt tiêu được bọng nhiễu tĩnh lặng.\n\n")
    
    f.write("### B. Kiểm Định Chuẩn Hóa Jarque-Bera & Kurtosis\n")
    f.write("- **Ý nghĩa:** Trong tài chính, Time Bars nổi tiếng là tạo ra thị trường có Đuôi Béo (Fat Tails) tức là hay xuất hiện cực đoan, không giống phân phối chuẩn Gauss (chuông), làm các mô hình ML tuyến tính, GARCH hay NN bị 'choáng' mồi.\n")
    f.write("- **Kết quả mong đợi:** Bar nào có điểm Jarque-Bera Stat càng thấp (hoặc Kurtosis càng gần 3) nghĩa là phân phối lợi suất của nó tròn trịa nhất, dễ học nhất cho AI.\n\n")
    
    f.write("### C. Độ lệch chuẩn Số Lượng Bar / Tuần\n")
    f.write("- **Ý nghĩa:** Time Bars thì cứ đúng tíc tắc là vẽ, không quan tâm TT câm điếc hay nổ bùng. Nhưng Data-driven Bars vẽ nến dựa trên tốc độ bơm thông tin (Rate of flow). \n")
    f.write("- **Kết quả mong đợi:** Với Time bars, con số độ lệch này thường gần bằng 0 cực kỳ ổn định. Với Dollar/Imbalance, con số này phải dao động rất cao (Vì tuần yên tĩnh ko vẽ bar nào, tuần có tin ra vẽ hàng chục Bar). **Biến động số nến cao là bản chất ưu việt để thanh 'cắn' tin tức**.\n")

print("4. Đang vẽ biểu đồ báo cáo...")
plt.style.use('dark_background')

# Plot 1: Histogram Lợi suất
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Hình 1: Phân Phối Lợi Suất Của Các Loại Bars (Returns KDE)', fontsize=16, color='white')

for i, (name, bars) in enumerate(brands.items()):
    if bars.empty:
        continue
    ret = np.log(bars['close'] / bars['close'].shift(1)).dropna()
    sns.histplot(ret, bins=50, kde=True, ax=axes[i], color='cyan', stat='density')
    axes[i].set_title(f"{name}\nKurtosis: {stats.kurtosis(ret):.2f}")
    axes[i].set_xlabel('Log Lợi Suất')
    axes[i].set_ylabel('Mật Độ')

plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'returns_distribution.png'), dpi=300, facecolor=fig.get_facecolor(), bbox_inches='tight')
plt.close()

# Plot 2: Số Lượng Bar Mỗi Tuần (Trực quan hóa sự co giãn thông tin)
fig2 = plt.figure(figsize=(12, 6))
for name, bars in brands.items():
    if not bars.empty:
        weekly = bars.resample('W').size()
        plt.plot(weekly.index, weekly.values, marker='o', label=name, alpha=0.8, linewidth=2)

plt.title('Hình 2: Số Lượng Thanh Bar Chốt Sổ Mỗi Tuần (Rate of Information Flow)', fontsize=14)
plt.xlabel('Trục Thời Gian Thời Tiết')
plt.ylabel('Khối Lượng Bar (Frequency)')
plt.legend()
plt.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'weekly_bars_count.png'), dpi=300, facecolor=fig2.get_facecolor(), bbox_inches='tight')
plt.close()

print("Hoàn tất! Báo cáo đã lưu tại thư mục:", out_dir)
