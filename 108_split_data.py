# 定義輸入與輸出檔案路徑
input_file_path = '108_year.xml'
output_file_path = '108_new_year.xml'

# 0x1B 在 Python 中表示為 '\x1b'
invalid_char = '\x1b'

print("正在清洗 XML 檔案中的無效字元...")

with open(input_file_path, 'r', encoding='utf-8', errors='ignore') as infile, \
     open(output_file_path, 'w', encoding='utf-8') as outfile:
    
    for line_num, line in enumerate(infile, 1):
        if invalid_char in line:
            # 將 0x1B 替換成空字串（刪除）
            line = line.replace(invalid_char, '')
            print(f"已清除第 {line_num} 行的無效字元 (0x1B)")
        outfile.write(line)

print("清洗完成！請嘗試使用『清洗後的檔案.xml』重新進行 CSV 轉換。")