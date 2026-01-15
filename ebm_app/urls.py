from django.urls import path
from . import views

urlpatterns = [
    path('', views.home_view, name='home'),
    # 🟢 這行要改：加入 <str:patient_id>（或 int 也可以，但要跟 view 參數一致）
    path('dashboard/<str:patient_id>/', views.dashboard_view, name='dashboard'),

    # 其他保持不變
    path('ajax/global_explanation/', views.ajax_global_explanation, name='ajax_global_explanation'),
    path('ajax/local_explanation/<str:patient_id>/', views.ajax_local_explanation, name='ajax_local_explanation'),
    path('ajax/ai_explain_feature/', views.ajax_ai_explain_feature, name='ajax_ai_explain_feature'), #新增
    
    path(
        'api/generate_patient_report/<str:patient_id>/',
        views.generate_patient_report_api,
        name='generate_patient_report_api'
    ),

    # # （可選）報告頁
    # path(
    #     'patient_report/<str:patient_id>/',
    #     views.patient_report_view,
    #     name='patient_report'
    # ),
]