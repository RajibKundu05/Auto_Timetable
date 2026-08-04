# 🤖 AI Timetable Generator

An AI-powered automatic timetable generator built using **Python**, **Machine Learning**, and **Google OR-Tools**. The project generates conflict-free school timetables by combining a trained teacher prediction model with constraint optimization.

---

## 📌 Features

* 🧠 AI-based teacher prediction using a trained Random Forest model
* 📅 Automatic conflict-free timetable generation
* 👨‍🏫 Prevents teacher scheduling conflicts
* 📚 Assigns subjects according to required weekly hours
* 🏫 Supports multiple classes and sections
* 📊 Exports the generated timetable directly to a formatted Excel workbook
* ⚡ Built using Google OR-Tools CP-SAT Solver for optimization
* 🔧 Modular and easy to extend

---

## 🛠️ Tech Stack

* Python 3.x
* Pandas
* Scikit-learn
* Joblib
* Google OR-Tools
* OpenPyXL

---

## 📂 Project Structure

```text
AI_Timetable_Generator/
│
├── data/
│   ├── Book1.csv
│   └── teacher_dataset.csv
│
├── models/
│   └── predict_teacher_model.pkl
│
├── train_teacher_model.py
├── timetable_generator.py
├── export_excel.py
├── test_teacher_model.py
├── requirements.txt
├── README.md
└── generated_timetable.xlsx
```

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/AI_Timetable_Generator.git
cd AI_Timetable_Generator
```

Create a virtual environment (optional but recommended):

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

---

## 📦 Required Libraries

```text
pandas
scikit-learn
joblib
ortools
openpyxl
```

or install manually:

```bash
pip install pandas scikit-learn joblib ortools openpyxl
```

---

## 🚀 Training the AI Model

Train the teacher prediction model:

```bash
python train_teacher_model.py
```

This generates:

```text
models/predict_teacher_model.pkl
```

---

## ▶️ Generate Timetable

Run:

```bash
python timetable_generator.py
```

The program will:

* Load the trained AI model
* Predict suitable teachers
* Generate a conflict-free timetable
* Export the timetable as an Excel workbook

Output:

```text
generated_timetable.xlsx
```

---

## 🧠 How It Works

1. Historical timetable data is used to train a Random Forest classifier.
2. The trained model predicts the most suitable teacher for a given:

   * Subject
   * Day
   * Time slot
   * Class
   * Section
3. Google OR-Tools uses these predictions as preference scores while enforcing hard constraints such as:

   * No teacher clashes
   * One subject per class per period
   * Weekly subject hour requirements
   * Room allocation (if enabled)
4. The final optimized timetable is exported to Excel.

---

## 📊 Example Excel Output

Each worksheet represents a class.

| Period | Monday            | Tuesday           | Wednesday             | Thursday          | Friday                |
| ------ | ----------------- | ----------------- | --------------------- | ----------------- | --------------------- |
| 1      | Math (Mr. Sharma) | English (Ms. Sen) | Science (Mr. Das)     | History (Mr. Roy) | Computer (Mrs. Gupta) |
| 2      | English (Ms. Sen) | Math (Mr. Sharma) | Computer (Mrs. Gupta) | Science (Mr. Das) | Free                  |

---

## 📈 Future Improvements

* Teacher leave management
* Teacher workload balancing
* Laboratory scheduling
* Smart room allocation
* Elective subject support
* Automatic lunch break scheduling
* Genetic Algorithm optimization
* Web dashboard using Flask or FastAPI
* React frontend
* PDF timetable export
* Student timetable generation
* Teacher-wise timetable generation

---

## 🤝 Contributing

Contributions are welcome.

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes.
4. Open a Pull Request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

Developed as an AI-powered timetable generation project using Machine Learning and Constraint Programming.
