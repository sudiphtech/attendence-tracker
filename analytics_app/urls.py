from django.urls import path
from .views import (
    chatbot_api,
    chatbot_page,
    dashboard,
    import_all,
    import_attendance_csv,
    import_csv,
    import_students,
    predict_form,
    predict_risk,
    students_api,
)

urlpatterns = [
    path('chatbot/', chatbot_page, name='chatbot_page'),
    path('api/chatbot/', chatbot_api, name='chatbot_api'),
    path('predict/', predict_risk, name='predict_risk'),
    path('api/students/', students_api, name='students_api'),
    path('dashboard/', dashboard, name='dashboard'),
    path('predict-form/', predict_form, name='predict_form'),
    path('import-students/', import_students, name='import_students'),
    path('import-csv/', import_csv, name='import_csv'),
    path('import-attendance-csv/', import_attendance_csv, name='import_attendance_csv'),
    path('import-all/', import_all, name='import_all'),
]
