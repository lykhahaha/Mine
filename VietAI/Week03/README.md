
# Lab 1: Thực hành về Numpy và TensorFlow

Cấu trúc bài lab của mình ngày hôm nay sẽ gồm 2 phần chính. Phần đầu sẽ tập trung việc xây dựng hàm giả thuyết qua mô hình hồi quy tuyến tính với numpy. Phần sau sẽ chuyển qua viết bằng Tensorflow qua chương trình theo khung được cho ra trước.

```
conda create -n ai.course_lab1
source activate ai.course_lab1
conda install -c conda-forge tensorflow
pip install matplotlib
pip install pandas
conda install jupyter
echo 'done'
jupyter notebook
```



## PHẦN 1: Numpy

### Numpy là gì và tại sao Numpy là gì?

Là thư viện Python cho phép thao tác trên mảng đa chiều, và các đối tượng có nguồn gốc khác nhau (như các mảng và ma trận khác) qua các phép toán, logic, xử lý, phân loại, lựa chọn, đọc và xuất, Biến đổi Fourier rời rạc, đại số tuyến tính cơ bản, hoạt động thống kê cơ bản, mô phỏng ngẫu nhiên và nhiều hơn nữa.

* ** Nhanh chóng **:
     * Numpy được viết bằng `C`
     * Một mã được viết bằng python bình thường có thể mất đến 30 phút, nhưng chỉ có 2,5`s` với `numpy`


* ** Tối ưu bộ nhớ **:
     * Bằng cách tạo kích thước cố định khi tạo, không giống như các danh sách `Python` (có thể phát triển tự động). Thay đổi kích thước của một `ndarray` sẽ tạo ra một mảng mới và xóa bản gốc.
     * Các phần tử trong một mảng `NumPy` đều phải có cùng kiểu dữ liệu và do đó sẽ có kích thước giống nhau trong bộ nhớ.


## PHẦN 2: Tensorflow

### Tensorflow

* Là ** thư viện phần mềm nguồn mở ** do Google phát triển, đặc biệt là cho Nghiên cứu về Nghiên cứu sâu vào năm 2015. Phiên bản hiện tại là `1.3`

* Bằng cách sử dụng một khái niệm **đồ thị tính toán** (computational graph) với ** các nút ** (node) như các phép toán và ** cạnh ** (edge) như là một mảng kích thước đa chiều.

* `Tensorflow` được triển khai với CPU, GPU và TPU trên các thiết bị như điện thoại di động, máy tính để bàn hoặc máy chủ với một API duy nhất.


### Hướng dẫn cài đặt

* Vui lòng vào liên kết bên dưới để cài đặt

> `https://github.com/tensorflow/tensorflow`

### 'Hello world!'

* Để kiểm tra liệu thiết lập của bạn đã thành công, hãy thử thực hiện đoạn mã sau:

```
import tensorflow as tf

c = tf.constant("Hello, distributed TensorFlow!")

sess = tf.Session()
sess.run(c)
```
