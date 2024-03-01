import re
def preprocessing(file):
    with open(file,'r', encoding='utf-8') as f:
        full_text = f.read()
    full_text.replace("المملكة المغربية\nوزارة العدل\nمديرية التشريع\n","")
    pattern = r"\nالمادة \d+\n"
    matches = re.finditer(pattern , full_text)
    chapter_contents = []
    start_index = 0
    for match in matches:
        end_index = match.start()
        chapter_content = full_text[start_index:end_index].strip()
        chapter_contents.append(chapter_content)
        
        start_index = match.end()
    last_chapter_content = full_text[start_index:].strip()
    chapter_contents.append(last_chapter_content)
    data = []
    for i, content in enumerate(chapter_contents, start=1):
        d={"text":"","metadata":""}
        metadata =f"المادة رقم {i-1} "
        text = content
        d["text"] = text
        d["metadata"] = metadata
        data.append(d)
    data = data[1:]
    return data