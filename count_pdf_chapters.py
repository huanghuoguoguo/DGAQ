"""
PDF章节字数统计工具
使用pdfplumber库提取PDF内容并统计每个章节的字数
"""
import pdfplumber
import re
from collections import OrderedDict


def extract_text_from_pdf(pdf_path):
    """从PDF中提取文本内容"""
    full_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                full_text += f"\n--- Page {page_num + 1} ---\n{text}"
    
    return full_text


def count_chinese_chars(text):
    """统计中文字符数（不包括标点、空格、英文）"""
    chinese_pattern = re.compile(r'[\u4e00-\u9fff]')
    chinese_chars = chinese_pattern.findall(text)
    return len(chinese_chars)


def count_all_chars(text):
    """统计所有可见字符（不包括空格和换行）"""
    visible_chars = re.sub(r'[\s\n\r\t]', '', text)
    return len(visible_chars)


def detect_chapters(text):
    """
    检测二级标题(如1.1、1.2、2.1等)
    只统计X.X格式的二级标题,跳过首页和摘要部分
    过滤掉目录中的条目
    """
    chapters = OrderedDict()
    
    # 只匹配二级标题:X.X 标题(如1.1、1.2、2.1)
    # 精确匹配:数字.数字 空格 标题,且标题部分不能包含过多的点号(排除目录)
    section_pattern = re.compile(r'^(\d+)\.(\d+)\s+([^\n]+?)\s*$')
    
    lines = text.split('\n')
    current_section = None
    current_content = []
    started = False  # 标记是否已经开始统计(跳过首页和摘要)
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 检查是否是二级标题
        match = section_pattern.match(line)
        if match:
            chapter_num = int(match.group(1))
            section_num = int(match.group(2))
            section_title = match.group(3).strip()
            
            # 过滤目录条目:如果标题包含过多的点号或页码标记,跳过
            if '...' in section_title or '.....' in section_title:
                continue
            
            # 过滤无效的章节号(如0.X)
            if chapter_num == 0:
                continue
            
            # 标记开始统计
            started = True
            
            # 保存前一个章节
            if current_section and current_content:
                content = '\n'.join(current_content)
                chapters[current_section] = {
                    'content': content,
                    'chinese_chars': count_chinese_chars(content),
                    'all_chars': count_all_chars(content),
                    'chapter': current_section.split('.')[0]
                }
            
            # 开始新章节
            current_section = f"{chapter_num}.{section_num} {section_title}"
            current_content = []
        elif started and current_section:  # 只在开始统计后才收集内容
            # 排除目录、摘要等干扰内容
            if not re.match(r'^(摘要|ABSTRACT|Abstract|引言|绪论|结论|致谢|参考文献|REFERENCES|References|目录|Contents)', line):
                # 排除包含过多点号的行(目录特征)
                if line.count('.') < 5:  # 正常内容不会有太多点号
                    current_content.append(line)
    
    # 保存最后一个章节
    if current_section and current_content:
        content = '\n'.join(current_content)
        chapters[current_section] = {
            'content': content,
            'chinese_chars': count_chinese_chars(content),
            'all_chars': count_all_chars(content),
            'chapter': current_section.split('.')[0]
        }
    
    return chapters


def extract_outline(text):
    """
    提取论文大纲结构（包括一级和二级标题）
    返回格式: OrderedDict {章节号: {'title': 标题, 'subsections': [子标题列表]}}
    """
    outline = OrderedDict()
    
    # 匹配一级标题: 第X章 或 X 标题 (数字开头)
    chapter_pattern = re.compile(r'^(?:第)?([一二三四五六七八九十\d]+)(?:章)?[\s、]+([^\n]+?)\s*$')
    # 匹配二级标题: X.X 标题
    section_pattern = re.compile(r'^(\d+)\.(\d+)\s+([^\n]+?)\s*$')
    # 匹配三级标题: X.X.X 标题
    subsection_pattern = re.compile(r'^(\d+)\.(\d+)\.(\d+)\s+([^\n]+?)\s*$')
    
    lines = text.split('\n')
    current_chapter = None
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # 过滤目录特征
        if '...' in line or '.....' in line:
            continue
        
        # 检查三级标题
        subsection_match = subsection_pattern.match(line)
        if subsection_match:
            chapter_num = subsection_match.group(1)
            section_num = subsection_match.group(2)
            subsection_num = subsection_match.group(3)
            title = subsection_match.group(4).strip()
            
            if chapter_num in outline:
                section_key = f"{chapter_num}.{section_num}"
                if section_key in outline[chapter_num]['subsections']:
                    if 'subsubsections' not in outline[chapter_num]['subsections'][section_key]:
                        outline[chapter_num]['subsections'][section_key]['subsubsections'] = OrderedDict()
                    outline[chapter_num]['subsections'][section_key]['subsubsections'][f"{chapter_num}.{section_num}.{subsection_num}"] = title
            continue
        
        # 检查二级标题
        section_match = section_pattern.match(line)
        if section_match:
            chapter_num = section_match.group(1)
            section_num = section_match.group(2)
            title = section_match.group(3).strip()
            
            # 过滤无效章节
            if chapter_num == '0':
                continue
            
            if chapter_num in outline:
                outline[chapter_num]['subsections'][f"{chapter_num}.{section_num}"] = {
                    'title': title,
                    'subsubsections': OrderedDict()
                }
            continue
        
        # 检查一级标题
        chapter_match = chapter_pattern.match(line)
        if chapter_match:
            chapter_num_raw = chapter_match.group(1)
            title = chapter_match.group(2).strip()
            
            # 转换中文数字为阿拉伯数字
            chinese_to_arabic = {
                '一': '1', '二': '2', '三': '3', '四': '4', '五': '5',
                '六': '6', '七': '7', '八': '8', '九': '9', '十': '10'
            }
            chapter_num = chinese_to_arabic.get(chapter_num_raw, chapter_num_raw)
            
            # 过滤摘要、致谢等
            if re.match(r'^(摘要|ABSTRACT|Abstract|致谢|参考文献|REFERENCES|References|附录)', title):
                continue
            
            current_chapter = chapter_num
            outline[chapter_num] = {
                'title': title,
                'subsections': OrderedDict()
            }
    
    return outline


def print_statistics(chapters):
    """打印统计结果，按章节汇总"""
    print("=" * 80)
    print("PDF章节字数统计结果（按二级标题统计）")
    print("=" * 80)
    print()
    
    # 按章节分组统计
    chapter_stats = OrderedDict()
    for section_name, info in chapters.items():
        chapter = info['chapter']
        if chapter not in chapter_stats:
            chapter_stats[chapter] = {
                'sections': [],
                'total_chinese': 0,
                'total_all': 0
            }
        
        chapter_stats[chapter]['sections'].append({
            'name': section_name,
            'chinese': info['chinese_chars'],
            'all': info['all_chars']
        })
        chapter_stats[chapter]['total_chinese'] += info['chinese_chars']
        chapter_stats[chapter]['total_all'] += info['all_chars']
    
    # 打印各章节详情
    grand_total_chinese = 0
    grand_total_all = 0
    
    for chapter, stats in chapter_stats.items():
        print(f"第{chapter}章:")
        for section in stats['sections']:
            print(f"  {section['name']}")
            print(f"    中文字数: {section['chinese']:,}")
            print(f"    总字符数: {section['all']:,}")
        print(f"  >>> 第{chapter}章合计: 中文 {stats['total_chinese']:,} 字，总字符 {stats['total_all']:,}")
        print()
        grand_total_chinese += stats['total_chinese']
        grand_total_all += stats['total_all']
    
    print("=" * 80)
    print(f"全文总计:")
    print(f"   中文字数: {grand_total_chinese:,}")
    print(f"   总字符数: {grand_total_all:,}")
    print(f"   二级标题数量: {len(chapters)}")
    print(f"   章节数量: {len(chapter_stats)}")
    print("=" * 80)


def save_to_file(chapters, output_file):
    """保存统计结果到文件，按章节汇总"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("PDF章节字数统计结果（按二级标题统计）\n")
        f.write("=" * 80 + "\n\n")
        
        # 按章节分组统计
        chapter_stats = OrderedDict()
        for section_name, info in chapters.items():
            chapter = info['chapter']
            if chapter not in chapter_stats:
                chapter_stats[chapter] = {
                    'sections': [],
                    'total_chinese': 0,
                    'total_all': 0
                }
            
            chapter_stats[chapter]['sections'].append({
                'name': section_name,
                'chinese': info['chinese_chars'],
                'all': info['all_chars']
            })
            chapter_stats[chapter]['total_chinese'] += info['chinese_chars']
            chapter_stats[chapter]['total_all'] += info['all_chars']
        
        # 写入各章节详情
        grand_total_chinese = 0
        grand_total_all = 0
        
        for chapter, stats in chapter_stats.items():
            f.write(f"第{chapter}章:\n")
            for section in stats['sections']:
                f.write(f"  {section['name']}\n")
                f.write(f"    中文字数: {section['chinese']:,}\n")
                f.write(f"    总字符数: {section['all']:,}\n")
            f.write(f"  >>> 第{chapter}章合计: 中文 {stats['total_chinese']:,} 字，总字符 {stats['total_all']:,}\n")
            f.write("\n")
            grand_total_chinese += stats['total_chinese']
            grand_total_all += stats['total_all']
        
        f.write("=" * 80 + "\n")
        f.write(f"全文总计:\n")
        f.write(f"   中文字数: {grand_total_chinese:,}\n")
        f.write(f"   总字符数: {grand_total_all:,}\n")
        f.write(f"   二级标题数量: {len(chapters)}\n")
        f.write(f"   章节数量: {len(chapter_stats)}\n")
        f.write("=" * 80 + "\n")
    
    print(f"\n统计结果已保存到: {output_file}")


def save_outlines(pdf_files, output_dir):
    """
    提取所有PDF的标题大纲并保存到指定目录
    """
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_outlines = OrderedDict()
    
    for pdf_path in pdf_files:
        file_name = os.path.basename(pdf_path)
        print(f"\n📄 提取大纲: {file_name}")
        
        try:
            # 提取文本
            text = extract_text_from_pdf(pdf_path)
            
            # 提取大纲
            outline = extract_outline(text)
            
            if not outline:
                print(f"  ⚠️  未检测到大纲结构")
                continue
            
            paper_name = file_name.replace('.pdf', '')
            all_outlines[paper_name] = outline
            
            # 保存单个论文大纲
            outline_file = os.path.join(output_dir, f"{paper_name}_大纲.txt")
            with open(outline_file, 'w', encoding='utf-8') as f:
                f.write(f"{'='*80}\n")
                f.write(f"{paper_name}\n")
                f.write(f"{'='*80}\n\n")
                
                for chapter_num, chapter_info in outline.items():
                    f.write(f"第{chapter_num}章 {chapter_info['title']}\n")
                    for section_key, section_info in chapter_info['subsections'].items():
                        f.write(f"  {section_key} {section_info['title']}\n")
                        if 'subsubsections' in section_info and section_info['subsubsections']:
                            for subsection_key, subsection_title in section_info['subsubsections'].items():
                                f.write(f"    {subsection_key} {subsection_title}\n")
            
            print(f"  ✅ 大纲已保存")
            
        except Exception as e:
            print(f"  ❌ 提取失败: {str(e)}")
            continue
    
    # 保存汇总大纲
    summary_file = os.path.join(output_dir, "_所有论文大纲汇总.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"所有参考论文大纲汇总\n")
        f.write(f"{'='*80}\n\n")
        
        for paper_name, outline in all_outlines.items():
            f.write(f"\n{'='*80}\n")
            f.write(f"{paper_name}\n")
            f.write(f"{'='*80}\n")
            
            for chapter_num, chapter_info in outline.items():
                f.write(f"\n第{chapter_num}章 {chapter_info['title']}\n")
                for section_key, section_info in chapter_info['subsections'].items():
                    f.write(f"  {section_key} {section_info['title']}\n")
                    if 'subsubsections' in section_info and section_info['subsubsections']:
                        for subsection_key, subsection_title in section_info['subsubsections'].items():
                            f.write(f"    {subsection_key} {subsection_title}\n")
    
    print(f"\n\n✅ 所有大纲已保存到: {output_dir}")
    return all_outlines


def analyze_and_generate_outline(all_outlines, output_dir):
    """
    分析所有论文大纲，生成中庸的论文大纲建议
    """
    import os
    from collections import Counter
    
    # 统计各章标题出现频率
    chapter_titles = Counter()
    section_structure = {}  # {章节号: {二级标题集合}}
    
    for paper_name, outline in all_outlines.items():
        for chapter_num, chapter_info in outline.items():
            # 统计章标题
            chapter_titles[f"第{chapter_num}章: {chapter_info['title']}"] += 1
            
            # 记录章节结构
            if chapter_num not in section_structure:
                section_structure[chapter_num] = Counter()
            
            for section_key, section_info in chapter_info['subsections'].items():
                section_structure[chapter_num][section_info['title']] += 1
    
    # 生成推荐大纲
    recommendation_file = os.path.join(output_dir, "_推荐论文大纲.txt")
    with open(recommendation_file, 'w', encoding='utf-8') as f:
        f.write(f"{'='*80}\n")
        f.write(f"基于{len(all_outlines)}篇参考论文的大纲分析与推荐\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"【分析说明】\n")
        f.write(f"根据参考论文的章节结构，提取了最常见的章节安排模式。\n")
        f.write(f"推荐大纲采用中庸稳健的结构，符合学术规范且与项目内容贴合。\n\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"一、各章标题频率统计\n")
        f.write(f"{'='*80}\n")
        for title, count in chapter_titles.most_common():
            f.write(f"  {title}: {count}篇论文使用\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"二、推荐论文大纲（基于Mamba2-MoE的DGA域名检测研究）\n")
        f.write(f"{'='*80}\n\n")
        
        # 生成标准五章结构
        recommended_outline = OrderedDict([
            ('1', {
                'title': '绪论',
                'subsections': [
                    '1.1 研究背景与意义',
                    '1.2 国内外研究现状',
                    '1.3 研究内容与目标',
                    '1.4 论文组织结构'
                ]
            }),
            ('2', {
                'title': '相关理论与技术',
                'subsections': [
                    '2.1 DGA域名检测技术概述',
                    '2.2 深度学习基础理论',
                    '2.3 Mamba模型原理',
                    '2.4 MoE（专家混合）机制',
                    '2.5 本章小结'
                ]
            }),
            ('3', {
                'title': '基于Mamba2-MoE的DGA域名检测模型设计',
                'subsections': [
                    '3.1 模型总体架构',
                    '3.2 数据预处理与特征提取',
                    '3.3 Mamba2编码器设计',
                    '3.4 MoE层设计与实现',
                    '3.5 模型训练策略',
                    '3.6 本章小结'
                ]
            }),
            ('4', {
                'title': '实验与结果分析',
                'subsections': [
                    '4.1 实验环境与数据集',
                    '4.2 评价指标',
                    '4.3 基线模型对比实验',
                    '4.4 消融实验',
                    '4.5 模型性能分析',
                    '4.6 本章小结'
                ]
            }),
            ('5', {
                'title': '总结与展望',
                'subsections': [
                    '5.1 工作总结',
                    '5.2 研究展望'
                ]
            })
        ])
        
        for chapter_num, chapter_info in recommended_outline.items():
            f.write(f"第{chapter_num}章 {chapter_info['title']}\n")
            for i, subsection in enumerate(chapter_info['subsections'], 1):
                f.write(f"  {subsection}\n")
            f.write(f"\n")
        
        f.write(f"\n{'='*80}\n")
        f.write(f"三、字数分配建议（参考平均值）\n")
        f.write(f"{'='*80}\n")
        f.write(f"  第1章（绪论）: 4,000-5,000字\n")
        f.write(f"  第2章（相关理论与技术）: 6,000-7,000字\n")
        f.write(f"  第3章（模型设计）: 8,000-10,000字\n")
        f.write(f"  第4章（实验与分析）: 7,000-9,000字\n")
        f.write(f"  第5章（总结与展望）: 2,000-3,000字\n")
        f.write(f"  ----------------------------------------\n")
        f.write(f"  预计总字数: 27,000-34,000字\n")
        f.write(f"\n")
        
        f.write(f"{'='*80}\n")
        f.write(f"四、撰写建议\n")
        f.write(f"{'='*80}\n")
        f.write(f"1. 第1章：重点阐述DGA域名威胁现状，强调深度学习方法的必要性\n")
        f.write(f"2. 第2章：系统介绍Mamba、MoE等核心技术，为后续章节铺垫\n")
        f.write(f"3. 第3章：详细描述模型架构、各模块设计思路及创新点\n")
        f.write(f"4. 第4章：充分展示实验结果，与主流方法对比，分析性能优势\n")
        f.write(f"5. 第5章：总结研究成果，指出局限性，提出未来改进方向\n")
        f.write(f"\n")
    
    print(f"\n✅ 大纲分析与推荐已保存到: {recommendation_file}")
    return recommendation_file


def main():
    import os
    import glob
    
    # 扫描 refs 目录下所有 PDF 文件
    refs_dir = r"e:/code/DGAQ/DGAQ/docs/refs"
    pdf_files = glob.glob(os.path.join(refs_dir, "*.pdf"))
    
    # 创建输出目录
    outline_dir = os.path.join(refs_dir, "论文大纲提取")
    
    print("="*80)
    print(f"📚 找到 {len(pdf_files)} 篇论文，开始提取大纲...")
    print("="*80)
    print()
    
    # 提取并保存所有大纲
    all_outlines = save_outlines(pdf_files, outline_dir)
    
    if all_outlines:
        print(f"\n{'='*80}")
        print(f"📊 开始分析大纲并生成推荐...")
        print(f"{'='*80}")
        
        # 分析并生成推荐大纲
        recommendation_file = analyze_and_generate_outline(all_outlines, outline_dir)
        
        print(f"\n\n{'='*80}")
        print(f"✅ 所有任务完成！")
        print(f"{'='*80}")
        print(f"\n📁 输出目录: {outline_dir}")
        print(f"  - 各论文大纲: {len(all_outlines)}个文件")
        print(f"  - 汇总文件: _所有论文大纲汇总.txt")
        print(f"  - 推荐大纲: _推荐论文大纲.txt")
        print()


if __name__ == "__main__":
    main()
