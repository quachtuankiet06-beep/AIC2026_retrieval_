import os
import re
import argparse
import numpy as np


def generate_kis_frames(A: int, B: int = None, target_count: int = 100, second_interval: tuple = None) -> list[int]:
    """
    Sinh danh sách frame theo logic phân bổ điểm số (Top 1 -> Top 100).
    """
    if B is None or A == B:
        return [A + 5 * i for i in range(target_count)]

    if A > B:
        A, B = B, A

    L = B - A
    stepT5 = L / 3.0
    stepT15 = L / 9.0

    selected = []
    used = set()

    def add_val(val):
        cand = int(round(val))
        while cand in used:
            cand += 5
        selected.append(cand)
        used.add(cand)
        return cand

    # 1. Top 1 (1.0đ)
    top1 = int(round((A + B) / 2.0))
    if top1 == A or top1 == B:
        top1_cand = top1
        while top1_cand == A or top1_cand == B:
            top1_cand += 5
        top1 = top1_cand

    selected.append(top1)
    used.add(top1)

    # 2. Top 2 - 5 (0.8đ): Bắt buộc chứa 2 đầu A và B
    t2 = A
    t3 = int(round(A + stepT5))
    t4 = int(round(A + 2 * stepT5))
    t5 = B

    top2_5_candidates = [t2, t3, t4, t5]
    for idx, cand in enumerate(top2_5_candidates):
        if idx == 0:
            selected.append(A)
            used.add(A)
        elif idx == 3:
            selected.append(B)
            used.add(B)
        else:
            c = cand
            while c in used or c == B:
                c += 1 if c < B else 5
            selected.append(c)
            used.add(c)

    # 3. Top 6 - 15 (0.6đ): 10 frames
    for k in range(10):
        add_val(A + k * stepT15)

    # 4. Top 16 - 20 (5 frames): Fill by Sys
    for p in np.linspace(A, B, 5):
        add_val(p)

    # 5. Top 21 - 50 (30 frames): Fill by Sys
    for p in np.linspace(A, B, 30):
        add_val(p)

    # 6. Top 51 - 100 (50 frames): Fill by Sys
    if second_interval and (B - A + 1) < 50:
        A2, B2 = second_interval
        for p in np.linspace(A2, B2, 50):
            add_val(p)
    else:
        for p in np.linspace(A, B, 50):
            add_val(p)

    while len(selected) < target_count:
        add_val(selected[-1] + 5)

    return selected[:target_count]


def parse_and_process_line(line: str, target_count: int = 100) -> list[str]:
    """
    Xử lý một dòng input và trả về danh sách các dòng kết quả (đủ 100 dòng).
    Hỗ trợ cả KIS và QA.
    """
    line = line.strip()
    if not line:
        return []

    # 1. Tìm video ID ở đầu
    vid_m = re.match(r'^\s*([A-Za-z0-9_]+)', line)
    if not vid_m:
        raise ValueError(f"Không tìm thấy Video ID ở đầu dòng: {line}")
    video = vid_m.group(1)
    rest = line[vid_m.end():].strip()
    if rest.startswith(','):
        rest = rest[1:].strip()

    # 2. Tìm khoảng [A - B] hoặc [A] hoặc số đơn A
    bracket_matches = re.findall(r'\[(\d+)\s*-\s*(\d+)\]', rest)
    single_bracket = re.findall(r'\[(\d+)\]', rest)

    intervals = []
    is_single = False
    answer = None

    last_bracket_idx = rest.rfind(']')
    if last_bracket_idx != -1:
        ans_part = rest[last_bracket_idx + 1:].strip()
        if ans_part.startswith(','):
            ans_part = ans_part[1:].strip()
        if ans_part:
            answer = ans_part
    else:
        num_m = re.search(r'^\s*(\d+)', rest)
        if num_m:
            intervals.append((int(num_m.group(1)), int(num_m.group(1))))
            is_single = True
            ans_part = rest[num_m.end():].strip()
            if ans_part.startswith(','):
                ans_part = ans_part[1:].strip()
            if ans_part:
                answer = ans_part

    if bracket_matches:
        for start, end in bracket_matches:
            intervals.append((int(start), int(end)))
    elif single_bracket and not intervals:
        intervals.append((int(single_bracket[0]), int(single_bracket[0])))
        is_single = True

    if not intervals:
        raise ValueError(f"Không nhận diện được frame từ dòng: {line}")

    A, B = intervals[0]
    second_interval = intervals[1] if len(intervals) > 1 else None

    frames = generate_kis_frames(A, B if not is_single else None, target_count, second_interval)

    # Chuẩn hóa answer: loại bỏ newline, quotes thừa và bọc ngoặc kép sạch
    if answer:
        answer = answer.replace('\r', '').replace('\n', ' ').strip().strip('"').strip("'").strip()
        answer = f'"{answer}"'

    result_lines = []
    for f in frames:
        if answer:
            result_lines.append(f"{video},{f},{answer}")
        else:
            result_lines.append(f"{video},{f}")

    return result_lines


def process_file(file_path: str, output_path: str = None, target_count: int = 100):
    """
    Đọc file CSV (KIS hoặc QA), biến đổi sang 100 dòng theo logic mới và lưu lại.
    """
    if output_path is None:
        output_path = file_path

    if not os.path.exists(file_path):
        print(f"File không tồn tại: {file_path}")
        return

    with open(file_path, 'r', encoding='utf-8-sig') as fp:
        raw_lines = [l.strip() for l in fp if l.strip()]

    if not raw_lines:
        print(f"File trống: {file_path}")
        return

    # Trường hợp 1: File định nghĩa dạng [A - B]
    if len(raw_lines) == 1 or '[' in raw_lines[0]:
        output_lines = parse_and_process_line(raw_lines[0], target_count)
    else:
        # Trường hợp 2: File đã có nhiều dòng cần refill lại
        first_line_parts = raw_lines[0].split(',', 2)
        video = first_line_parts[0].strip()
        answer = first_line_parts[2].strip() if len(first_line_parts) >= 3 else None

        frames = []
        for line in raw_lines:
            parts = line.split(',')
            if len(parts) >= 2:
                try:
                    frames.append(int(parts[1].strip()))
                except ValueError:
                    pass

        if not frames:
            print(f"Không tìm thấy frame hợp lệ trong: {file_path}")
            return

        A = frames[0]
        B = frames[-1]
        is_single = (A == B)

        new_frames = generate_kis_frames(A, B if not is_single else None, target_count)

        if answer:
            answer = answer.replace('\r', '').replace('\n', ' ').strip().strip('"').strip("'").strip()
            answer = f'"{answer}"'

        output_lines = []
        for f in new_frames:
            if answer:
                output_lines.append(f"{video},{f},{answer}")
            else:
                output_lines.append(f"{video},{f}")

    with open(output_path, 'w', encoding='utf-8', newline='\n') as fp:
        fp.write('\n'.join(output_lines) + '\n')

    print(f"Đã xử lý: {os.path.basename(file_path)} -> Đã ghi {len(output_lines)} dòng vào {os.path.basename(output_path)}.")


def process_directory(dir_path: str, target_count: int = 100):
    """Xử lý toàn bộ các file .csv (cả KIS và QA) trong một thư mục."""
    files = sorted([f for f in os.listdir(dir_path) if f.endswith('.csv')])
    for f in files:
        full_path = os.path.join(dir_path, f)
        process_file(full_path, target_count=target_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script tạo 100 dòng KIS và QA theo logic phân bổ điểm số.")
    parser.add_argument('--file', '-f', type=str, default='Test.csv', help='Đường dẫn tới file CSV cần xử lý (mặc định: Test.csv)')
    parser.add_argument('--dir', '-d', type=str, default=None, help='Đường dẫn thư mục chứa các file CSV cần xử lý')
    parser.add_argument('--input', '-i', nargs='+', type=str, help='Chuỗi input trực tiếp')
    parser.add_argument('--output', '-o', type=str, default='Test.csv', help='Đường dẫn file lưu kết quả (mặc định: Test.csv)')

    args = parser.parse_args()

    if args.input:
        input_str = ' '.join(args.input)
        lines = parse_and_process_line(input_str)
        out_file = args.output
        with open(out_file, 'w', encoding='utf-8', newline='\n') as fp:
            fp.write('\n'.join(lines) + '\n')
        print(f"Đã ghi {len(lines)} dòng vào file {out_file}")
        print("\n".join(lines[:5]))
        print(f"... (Tổng cộng {len(lines)} dòng)")
    elif args.dir:
        process_directory(args.dir)
    else:
        target_file = args.file if args.file else 'Test.csv'
        process_file(target_file, output_path=args.output)
