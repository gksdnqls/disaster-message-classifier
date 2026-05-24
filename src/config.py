LABEL_NAMES = ["일반", "주의", "긴급"]
LABEL_TO_ID = {"일반": 0, "주의": 1, "긴급": 2}
ID_TO_LABEL = {0: "일반", 1: "주의", 2: "긴급"}

TEXT_COLUMN = "메시지내용"
LABEL_COLUMN = "label_id"

TRAIN_CLASS_WEIGHTS = {
    0: 0.40402502390506395,
    1: 2.2883393685280478,
    2: 11.375572398547291,
}
