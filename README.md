# OAG-BE-ENHANCED

## Giới thiệu

OAG-BE-ENHANCED là một API tìm kiếm hình ảnh tương tự sử dụng OpenCV và Flask.

## Yêu cầu hệ thống

- Python 3.11 trở lên
- pip (Python package installer)

## Cài đặt

1. Clone repository:

    ```sh
    git clone https://github.com/OnlineArtGallery-SEP490-SP25-SE11/OAG-BE-ENHANCED.git ./api-enhanced
    cd api-enhanced
    ```

2. Tạo và kích hoạt môi trường ảo:

    ```sh
    python -m venv .venv
    source .venv/bin/activate  # Trên Windows sử dụng: .venv\Scripts\activate
    ```

3. Cài đặt các gói phụ thuộc:

    ```sh
    pip install -r requirements.txt
    ```

## Khởi chạy

1. Chạy ứng dụng Flask:

    ```sh
    python main.py
    ```

2. API sẽ chạy trên `http://localhost:8080`.
