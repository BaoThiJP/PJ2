import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

#GUI-------------------------
st.set_page_config(page_title="Hotel Recommendation System", layout="centered")

# Tạo thanh menu bên trái
menu = ["Trang Chủ", "Giới Thiệu Project", "Mô hình-Content-Based Filtering", "Mô hình-Collaborative Filtering","Dự Đoán Mới"]
choice = st.sidebar.selectbox('Menu', menu)

if choice == 'Trang Chủ':    
    # Thiết lập cấu hình trang
    
    st.title("Hotel Recommendation System")
    st.subheader("Khám Phá Những Điểm Lưu Trú Tốt Nhất Dành Cho Bạn")

    # Hiển thị hình ảnh
    st.image("hotel-recommendation-systems.jpg", caption="Tiện Nghi Đẳng Cấp – Trải Nghiệm Dịch Vụ Vượt Trội – Một Kỳ Nghỉ Thư Giãn Đúng Nghĩa", use_column_width=True)
    st.write(
        """
        Chào mừng bạn đến với ứng dụng gợi ý khách sạn của chúng tôi! Đây là một ứng dụng được thiết kế để giúp bạn tìm kiếm và lựa chọn những điểm lưu trú tốt nhất dựa trên sở thích và nhu cầu cá nhân của bạn.

        Sứ mệnh của chúng tôi hướng đến việc cung cấp những gợi ý phù hợp về khách sạn, resort, và các điểm lưu trú. Với công nghệ tiên tiến và phân tích dữ liệu thông minh, chúng tôi giúp bạn dễ dàng tìm được nơi nghỉ dưỡng lý tưởng cho kỳ nghỉ của bạn.

        **Lợi Ích Cho Người Dùng:**
        - **Gợi Ý Cá Nhân Hóa:** Nhận những đề xuất khách sạn phù hợp với sở thích và yêu cầu của bạn.
        - **Tiết Kiệm Thời Gian:** Tiết kiệm thời gian tìm kiếm
        - **Trải Nghiệm Tuyệt Vời:** Tận hưởng những dịch vụ và tiện nghi tốt nhất tại các điểm lưu trú hàng đầu, giúp kỳ nghỉ của bạn trở nên đáng nhớ hơn.

        Khám phá và tận hưởng kỳ nghỉ hoàn hảo với ứng dụng gợi ý khách sạn của chúng tôi!
        """)

elif choice == 'Giới Thiệu Project':  
    st.subheader("Giới Thiệu Project")
    st.write(
        """
        Mục tiêu của project nhằm xây dựng một ứng dụng thông minh giúp người dùng tìm kiếm và lựa chọn các sản phẩm hoặc dịch vụ phù hợp với sở thích và nhu cầu của khách hàng. 
        Hệ thống gợi ý sẽ sử dụng hai phương pháp phân tích dữ liệu chính để cung cấp các gợi ý cá nhân hóa, từ đó cải thiện trải nghiệm người dùng.

        """)

    with st.expander("Phương Pháp Xây Dựng Mô Hình"):
        st.write(
            """
            - **Content-Based Filtering**: Gợi ý dựa trên các đặc điểm của khách sạn như dịch vụ, tiện nghi, và loại phòng. Mô hình so sánh đặc điểm của khách sạn với sở thích người dùng để đưa ra các gợi ý tương tự.
            - **Collaborative Filtering**: Gợi ý dựa trên hành vi và sở thích của người dùng tương tự. Mô hình phân tích dữ liệu từ người dùng có sở thích giống nhau để đưa ra các gợi ý mới.
            """)
    
    st.image("https://www.researchgate.net/publication/337401660/figure/fig1/AS:827362874777603@1574270089942/The-principle-behind-collaborative-and-content-based-filtering-9-Pilah-Matur-App.ppm", use_column_width=True)



elif choice == 'Mô hình-Content-Based Filtering':
    import content_based







elif choice == 'Mô hình-Collaborative Filtering':
    import collaborative

# elif choice == 'Dự Đoán Mới':