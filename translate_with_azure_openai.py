"""
Script để dịch target_text từ tiếng Anh sang tiếng Việt sử dụng Azure OpenAI API
Model: GPT-4o-mini
"""

import pandas as pd
import os
from openai import AzureOpenAI
from tqdm import tqdm
import time
import json

# ==========================================
# 1. CẤU HÌNH AZURE OPENAI
# ==========================================
# Thay đổi các giá trị này theo cấu hình Azure của bạn
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2023-05-15")
DEPLOYMENT_NAME = "gpt-4o-mini"

# Khởi tạo Azure OpenAI client
client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    api_version=AZURE_OPENAI_API_VERSION,
    azure_endpoint=AZURE_OPENAI_ENDPOINT
)

# ==========================================
# 2. PROMPT DỊCH THUẬT
# ==========================================
TRANSLATION_PROMPT = """You are an expert English-to-Vietnamese translator focused on semantic accuracy and natural localization.

YOUR TASKS:
1. Translate the input text into natural, polite Vietnamese suitable for the context.
2. PRESERVE ALL TAGS inside brackets (e.g., [FIRSTNAME], [PHONEIMEI]) exactly as they appear. Do not translate, modify, or remove them.
3. ADJUST sentence structure to fit Vietnamese grammar while keeping tags in logical positions.

RULES:
- For administrative texts (IMEI, records): Use a professional, neutral tone.
- For personal communication (Dear [NAME]): Use a respectful, friendly tone (e.g., "Thân gửi", "Chào").
- Handling [AGE] and [GENDER]: In Vietnamese, adjectives often follow nouns. Adjust phrasing naturally (e.g., "male [AGE] old" -> "nam [AGE] tuổi").
- Output ONLY the translated text. No quotation marks, no explanations.

EXAMPLES:
Input: "Hello [FIRSTNAME], please check device [DEVICE_ID]."
Output: Chào [FIRSTNAME], vui lòng kiểm tra thiết bị [DEVICE_ID].

Input: "Review parameters for [GENDER] patient of [AGE]."
Output: Xem xét các thông số cho bệnh nhân [GENDER] [AGE] tuổi."""

# ==========================================
# 3. HÀM DỊCH THUẬT
# ==========================================
def translate_text(text, max_retries=3, delay=1):
    """
    Dịch text từ tiếng Anh sang tiếng Việt sử dụng Azure OpenAI
    
    Args:
        text: Text cần dịch
        max_retries: Số lần thử lại nếu lỗi
        delay: Thời gian chờ giữa các lần thử (giây)
    
    Returns:
        Translated text hoặc None nếu lỗi
    """
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).strip()
    if not text:
        return ''
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=DEPLOYMENT_NAME,
                messages=[
                    {"role": "system", "content": TRANSLATION_PROMPT},
                    {"role": "user", "content": f"Translate this text:\n\n{text}"}
                ],
                temperature=0.3,  # Thấp để đảm bảo tính nhất quán
                max_tokens=1000
            )
            
            translated = response.choices[0].message.content.strip()
            
            # Loại bỏ dấu ngoặc kép nếu có
            if translated.startswith('"') and translated.endswith('"'):
                translated = translated[1:-1]
            elif translated.startswith("'") and translated.endswith("'"):
                translated = translated[1:-1]
            
            return translated
            
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"⚠️ Lỗi khi dịch (thử lại {attempt + 1}/{max_retries}): {e}")
                time.sleep(delay * (attempt + 1))  # Exponential backoff
            else:
                print(f"❌ Lỗi khi dịch sau {max_retries} lần thử: {e}")
                return None
    
    return None

# ==========================================
# 4. XỬ LÝ FILE CSV
# ==========================================
def translate_csv(input_file, output_file=None, batch_size=10, start_from=0, max_rows=None):
    """
    Dịch tất cả target_text trong file CSV
    
    Args:
        input_file: Đường dẫn file CSV đầu vào
        output_file: Đường dẫn file CSV đầu ra (mặc định: thêm _translated)
        batch_size: Số dòng xử lý mỗi batch (để hiển thị progress)
        start_from: Bắt đầu từ dòng nào (để tiếp tục nếu bị gián đoạn)
        max_rows: Giới hạn số dòng xử lý (None = tất cả)
    """
    print("=" * 80)
    print("DỊCH THUẬT VỚI AZURE OPENAI")
    print("=" * 80)
    
    # Đọc file CSV
    print(f"\n[1/4] Đang đọc file: {input_file}")
    df = pd.read_csv(input_file, encoding='utf-8')
    total_rows = len(df)
    print(f"✓ Đã đọc {total_rows} dòng")
    
    # Kiểm tra cột target_text
    if 'target_text' not in df.columns:
        print("❌ Không tìm thấy cột 'target_text' trong file!")
        return
    
    # Giới hạn số dòng nếu cần
    if max_rows:
        df = df.head(max_rows)
        total_rows = len(df)
        print(f"✓ Giới hạn xử lý {total_rows} dòng")
    
    # Tạo cột mới cho kết quả dịch
    if 'target_text_vi' not in df.columns:
        df['target_text_vi'] = ''
    
    # Xử lý từ dòng start_from
    df_to_process = df.iloc[start_from:].copy()
    rows_to_process = len(df_to_process)
    
    print(f"\n[2/4] Bắt đầu dịch từ dòng {start_from + 1} đến {start_from + rows_to_process}")
    print(f"✓ Cần dịch {rows_to_process} dòng")
    
    # Dịch từng dòng
    print(f"\n[3/4] Đang dịch...")
    translated_count = 0
    error_count = 0
    
    for idx, row in tqdm(df_to_process.iterrows(), total=rows_to_process, desc="Dịch thuật"):
        original_text = row['target_text']
        
        # Kiểm tra xem đã dịch chưa
        if pd.notna(df.loc[idx, 'target_text_vi']) and df.loc[idx, 'target_text_vi'] != '':
            translated_count += 1
            continue
        # Dịch text
        translated = translate_text(original_text)
        
        if translated:
            df.loc[idx, 'target_text_vi'] = translated
            translated_count += 1
        else:
            error_count += 1
            # Giữ nguyên text gốc nếu lỗi
            df.loc[idx, 'target_text_vi'] = original_text
        
        # Lưu tạm mỗi batch_size dòng
        if (translated_count + error_count) % batch_size == 0:
            if output_file:
                df.to_csv(output_file, index=False, encoding='utf-8')
            print(f"  Đã dịch: {translated_count}/{rows_to_process} | Lỗi: {error_count}")
        
        # Thêm delay nhỏ để tránh rate limit
        time.sleep(0.1)
    
    # Xác định file output
    if output_file is None:
        base_name = os.path.splitext(input_file)[0]
        output_file = f"{base_name}_translated.csv"
    
    # Lưu kết quả
    print(f"\n[4/4] Đang lưu kết quả...")
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    print("\n" + "=" * 80)
    print("HOÀN THÀNH!")
    print("=" * 80)
    print(f"✓ Tổng số dòng: {total_rows}")
    print(f"✓ Đã dịch thành công: {translated_count}")
    print(f"✓ Số lỗi: {error_count}")
    print(f"✓ File kết quả: {output_file}")
    print("=" * 80)

# ==========================================
# 5. CHẠY SCRIPT
# ==========================================
if __name__ == "__main__":
    # Cấu hình
    INPUT_FILE = "data/pii_records.csv"
    OUTPUT_FILE = "data/pii_records_translated.csv"  # None để tự động tạo tên
    
    # Kiểm tra API key
    if AZURE_OPENAI_API_KEY == "your-api-key-here" or not AZURE_OPENAI_API_KEY:
        print("⚠️ CẢNH BÁO: Chưa cấu hình Azure OpenAI API Key!")
        print("\nCách cấu hình:")
        print("1. Đặt biến môi trường:")
        print("   - AZURE_OPENAI_ENDPOINT")
        print("   - AZURE_OPENAI_API_KEY")
        print("   - AZURE_OPENAI_API_VERSION (tùy chọn)")
        print("\n2. Hoặc sửa trực tiếp trong script:")
        print("   - AZURE_OPENAI_ENDPOINT = 'https://your-resource.openai.azure.com/'")
        print("   - AZURE_OPENAI_API_KEY = 'your-api-key'")
        print("\n3. Kiểm tra DEPLOYMENT_NAME phù hợp với deployment trên Azure")
        
        response = input("\nBạn có muốn tiếp tục? (y/n): ")
        if response.lower() != 'y':
            exit()
    
    # Chạy dịch thuật
    translate_csv(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        batch_size=100,  # Lưu tạm mỗi 50 dòng
        start_from=33170,   # Bắt đầu từ dòng 0
        max_rows=40000   # None = dịch tất cả, hoặc đặt số để test (ví dụ: 100)
    )

