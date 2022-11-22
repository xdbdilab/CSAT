# x264_python
- version: r2334
- Workload: any video(.mp4)
- Performance: video rate(kb/s)
- Optimization goal: max
- Number of options: 9/140(Tuning/Total)

# Prerequisites
- Python 3.x
- numpy 1.19.2
- pandas 1.1.5

# Installation
1. Download and install [Python 3.x](https://www.python.org/downloads/).

2. Install numpy

   ``` $ pip install numpy```

3. Install pandas

   ``` $ pip install pandas```

# 使用方法：
import x264.main as x264

# 配置（configuration）->性能
no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref = configuration
return x264.X264.getPerformance(no_8x8dct,no_cabac,no_deblock,no_fast_pskip,no_mbtree,no_mixed_refs,no_weightb,rc_lookahead,ref)

# 随机生成数据
x264.X264.x264_data(size = 1000, file_name = "Test.csv")

# 配置选项及对应范围
# option		type	bound[default]
# no-8x8dct		binary	{0,1}[0]
# no-cabac		binary	{0,1}[0]
# no-deblock	binary	{0,1}[0]
# no-fast-pskip	binary	{0,1}[0]
# no-mbtree		binary	{0,1}[0]
# no-mixed-refs	binary	{0,1}[0]
# no-weightb	binary	{0,1}[0]
# rc-lookahead	int		[40, 250][40]
# ref			int		[1,9][1]