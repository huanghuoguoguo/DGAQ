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


def main():
    file_name = "基于深度学习的DGA域名检测方法研究_王天宇.pdf"

    # PDF文件路径
    pdf_path = r"e:/code/DGAQ/DGAQ/docs/refs/" + file_name
    output_file = r"e:/code/DGAQ/DGAQ/docs/refs/" + file_name.replace(".pdf","_章节统计.txt")
    
    print(f"正在读取PDF文件: {pdf_path}")
    print("这可能需要一些时间...")
    print()
    
    # 提取文本
    text = extract_text_from_pdf(pdf_path)
    
    # 检测章节
    print("正在检测章节...")
    chapters = detect_chapters(text)
    
    # 打印统计结果
    print_statistics(chapters)
    
    # 保存到文件
    save_to_file(chapters, output_file)


if __name__ == "__main__":
    main()
