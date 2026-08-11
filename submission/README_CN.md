# IJHMT投稿文件

- `IJHMT_UPLOAD_ORDER_CN.md`：最终系统上传顺序和当前唯一需作者补充的信息。
- `cover_letter_IJHMT.md`：本论文专用投稿信，不复用氚释放论文的投稿信。
- `highlights.txt`：5条Highlights，每条不超过85个英文字符。
- `title_page.txt`：单独上传的题目、作者、单位和通讯邮箱。
- `CRediT_author_statement.md`：五位作者的贡献说明。
- `declaration_of_competing_interest.md`：无利益冲突声明。
- `acknowledgements.md`：基金和项目致谢。
- `declaration_of_generative_ai_use.md`：与正文一致的AI辅助使用声明；明确计算、
  数据、引用和最终文字由作者核验，生成式AI未用于制作或修改科学图像和数值结果。
- `../manuscript/main.tex`及相关图片：可编辑论文源文件。
- `../manuscript/main.pdf`：当前22页英文预览稿；正式瞬态模型结果完成后统一刷新。
- 本轮投稿默认不附单独的Supplementary PDF。网格、时间步、外部对照、独立装填和
  主要模型结果均放在正文；精确数据划分、完整参数表和生成程序随数据代码包提供。
- `../manuscript/supplement.tex`仅作为项目内备用源码保留。它目前可自动编译成4页的
  精简Supplementary PDF，只含预先登记的数据划分、旧无界输出诊断和网格/时间步
  敏感性。默认投稿仍不上传；只有编辑在投稿或返修阶段明确要求时才加入上传包。

图文摘要是IJHMT鼓励但不强制的文件。待正式温度云图完成后，再从同一份
OpenFOAM--模型场对比数据制作，避免提前使用尚未完成的模型结果。

正式打包时，程序会从LaTeX正文自动提取7条图注，生成独立可编辑文件
`Figure_captions.tex`。当前题名页还需要作者提供通讯电话号码；程序只接受作者确认的
国际格式号码，不会自行猜测。

本项目专用Zenodo DOI尚未生成。最终处理数据、两张模型结果图和复现文件齐全后，
由固定程序同时写回正文、投稿信和数据记录；不得复用上一篇氚释放论文的DOI。
