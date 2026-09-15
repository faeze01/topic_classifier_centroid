from pydantic import BaseModel, ConfigDict


class ClassifyRequest(BaseModel):
    body: str
    title: str
    post_id: str
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "title": "تپش قلب همیشه از قلب نیست",
                "body": (
                    "تپش قلب در بیشتر موارد خطرناک نیست اما غش، درد سینه و تنگی نفس "
                    "باید همان لحظه پیگیری شوند. نوار قلب همان روز اغلب طبیعی است چون "
                    "فقط چند ثانیه را ثبت می‌کند. فشار خون خانگی باید با پنج دقیقه "
                    "استراحت و دو بار اندازه‌گیری ثبت شود تا عدد معتبر باشد."
                ),
                "post_id": "1",
            }
        }
    )

class TopicResult(BaseModel):
    topic: str

class ClassifyResponse(BaseModel):
    post_id: str
    topics: list[TopicResult]