import os
import joblib
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from data.dataset import SVMSpectrogramDataset

# ======================
# ⚙️ 配置参数
# ======================

EXPERIMENT_NAME = "exp_baseline_1000samples"
EXPERIMENT_DIR = f"E:/AMR/DA/Projekt/bird_cls_cnn/projekt_svm/experiments/{EXPERIMENT_NAME}"
MODEL_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "models", "svm_model.joblib")
REPORT_SAVE_PATH = os.path.join(EXPERIMENT_DIR, "classification_report.txt")

TRAIN_CSV = "E:/AMR/DA/Projekt/data/data_list/0408/train_list_high_quality.csv"
VALID_CSV = "E:/AMR/DA/Projekt/data/data_list/0408/valid_list_high_quality.csv"
MAX_SAMPLES_PER_CLASS_TRAIN = 1000
MAX_SAMPLES_PER_CLASS_VALID = 300
PCA_COMPONENTS = 256
PCA_BATCH_SIZE = 256

MODEL_TYPE = "sgd"  # <<< 只需要在这里改成 "svm" 或 "sgd"

# ======================
# 📁 创建必要目录
# ======================
os.makedirs(os.path.join(EXPERIMENT_DIR, "models"), exist_ok=True)

# ======================
# 📥 加载训练集
# ======================
print("📥 加载训练集 + PCA 拟合...")
train_dataset = SVMSpectrogramDataset(
    csv_path=TRAIN_CSV,
    max_samples_per_class=MAX_SAMPLES_PER_CLASS_TRAIN,
    pca_components=256,
    pca_batch_size=256,
    pca_save_path="E:/AMR/DA/Projekt/bird_cls_cnn/projekt_svm/pca/pca_model_1000samples.joblib",
    use_saved_pca=False
)
X_train, y_train = train_dataset.get_data()
label_mapping = train_dataset.get_label_mapping()

# ======================
# 📥 加载测试集
# ======================
print("📥 加载测试集 + 使用训练集的 PCA...")
valid_dataset = SVMSpectrogramDataset(
    csv_path=VALID_CSV,
    label_mapping=train_dataset.get_label_mapping(),
    max_samples_per_class=MAX_SAMPLES_PER_CLASS_VALID,
    pca_components=256,
    pca_batch_size=256,
    pca_save_path="E:/AMR/DA/Projekt/bird_cls_cnn/projekt_svm/pca/pca_model_1000samples.joblib",
    use_saved_pca=True
)
X_test, y_test = valid_dataset.get_data()

# ======================
# 🧠 初始化模型
# ======================
if MODEL_TYPE == "svm":
    print("🚀 使用 LinearSVC 训练...")
    clf = LinearSVC(C=1.0, max_iter=5000)
elif MODEL_TYPE == "sgd":
    print("🚀 使用 SGDClassifier (hinge loss) 训练...")
    clf = SGDClassifier(loss="hinge", penalty="l2", alpha=1e-4, max_iter=1000, random_state=42)
else:
    raise ValueError(f"未知模型类型: {MODEL_TYPE}")

# ======================
# 🏋️‍♂️ 训练模型
# ======================
clf.fit(X_train, y_train)
print("✅ 模型训练完成")

# ======================
# 📈 评估模型
# ======================
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=label_mapping.keys())

print(f"\n🎯 测试集准确率: {acc:.4f}")
print("\n📄 分类报告:\n")
print(report)

# ======================
# 💾 保存模型和报告
# ======================
joblib.dump(clf, MODEL_SAVE_PATH)
print(f"✅ 模型已保存至: {MODEL_SAVE_PATH}")

with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
    f.write(f"测试集准确率: {acc:.4f}\n\n")
    f.write(report)
print(f"📝 分类报告已保存至: {REPORT_SAVE_PATH}")
