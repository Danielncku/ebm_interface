
from django.shortcuts import render
from django.http import HttpResponse,  JsonResponse
from .ml_models import MLInterpretModel
import os
os.environ["OPENAI_BASE_URL"] = "http://192.168.63.184:11434/v1"
os.environ["OPENAI_API_KEY"] = "ollama"
    
try:
    from t2ebm.llm import openai_setup
    from t2ebm import describe_graph
    
    # 用 Ollama 啟動
    llm = openai_setup(
        model="gpt-oss:20b",   # 你的 Ollama 模型
        
        base_url="http://192.168.63.184:11434/v1"
    )
    
    TALK_TO_EBM_AVAILABLE = True
    print("✅ TalkToEBM 已使用 Ollama 啟動")

except Exception as e:
    TALK_TO_EBM_AVAILABLE = False
    llm = None
    describe_graph = None
    print(f"❌ TalkToEBM 初始化失敗（已禁用）: {e}")
    
# 初始化模型
feature_cols = ["Sex", "DM", "HTN", "CAD", "Age", "Pre_HD_SBP", "HR", "RR", "blood-speed",
                "Dialysis-blood-temp", "Dialysis-blood-rate", "start-weight", "Mean_BP",
                "HR_Mean_BP", "UF_BW_perc", "透析液 Ca", "體溫_New", "預估脫水量",
                "靜脈壓(mmHg)", "透析液壓(mmHg)", 'idh_count_last_28d']
target_col = "Nadir90/100"

#patient data
# ml_model = MLInterpretModel("EBM_28.joblib", "Patient5.csv", feature_cols, target_col)
# API data
ml_model = MLInterpretModel("EBM_28.joblib", "interface/data/temp.csv", feature_cols, target_col)

# 首頁
def home_view(request):
    data = ml_model.data
    patient_list = data['ID'].unique().tolist()
    return render(request, 'ebm_app/home.html', {
        'patient_list': patient_list,
        'feature_cols': ml_model.feature_cols
    })

# Dashboard 頁面
def dashboard_view(request, patient_id):
    data = ml_model.data
    patient_list = data['ID'].unique().tolist()
    feature_cols = ml_model.feature_cols
    return render(request, 'ebm_app/dashboard.html', {
        'patient_id': patient_id,
        'patient_list': patient_list,
        'feature_cols': feature_cols
    })


# 全域解釋 AJAX
def ajax_global_explanation(request):
    feature = request.GET.get('feature', None)
    if feature == '':
        feature = None
    
    # 🆕 新增：讀取密度視窗參數
    density_enabled = request.GET.get('density_window', 'false').lower() == 'true'
    lower_percentile = float(request.GET.get('lower_percentile', 2.5))
    upper_percentile = float(request.GET.get('upper_percentile', 97.5))
    
    # 🆕 新增：傳遞參數給 ml_model
    html = ml_model.get_global_explanation_html(
        feature=feature,
        density_window=density_enabled,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile
    )
    return HttpResponse(html)

# ✅ 修改：區域解釋 AJAX 加入顯示模式參數
# 區域解釋 AJAX
def ajax_local_explanation(request, patient_id):
    # 🆕 新增：讀取顯示模式參數
    display_mode = request.GET.get('display_mode', 'all')
    html = ml_model.get_local_explanation_html(patient_id, display_mode)
    return HttpResponse(html)


# ============================================
# ✅ 加在 views.py 的最底部
# ============================================

def ajax_ai_explain_feature(request):
    """
    提供單一特徵的 AI 自然語言解釋
    """
    if not TALK_TO_EBM_AVAILABLE:
        return JsonResponse({
            'success': False,
            'error': 'TalkToEBM 未安裝'
        })
    
    feature_name = request.GET.get('feature')
    patient_id = request.GET.get('patient_id', None)
    
    if not feature_name:
        return JsonResponse({
            'success': False,
            'error': '請提供特徵名稱'
        })
    
    try:
        # 檢查特徵是否存在
        if feature_name not in ml_model.feature_cols:
            return JsonResponse({
                'success': False,
                'error': f'特徵 {feature_name} 不存在'
            })
        
        feature_idx = ml_model.feature_cols.index(feature_name)
        
        # 準備客製化 prompt
        custom_prompt = None
        patient_value = None
        
        if patient_id:
            patient_data = ml_model.data[ml_model.data['ID'] == str(patient_id)]
            if not patient_data.empty:
                patient_value = patient_data[feature_name].iloc[0]
                
                custom_prompt = (
                    f"角色：你是專業的透析醫療顧問。\n"
                    f"背景：病人的 {feature_name} = {patient_value:.2f}\n"
                    f"任務：\n"
                    f"1. 用簡單的話說明這個數值的意義\n"
                    f"2. 解釋它如何影響低血壓風險\n"
                    f"3. 給一個具體建議\n"
                    f"要求：\n"
                    f"- 不用專業術語\n"
                    f"- 語氣溫和鼓勵\n"
                    f"- 不超過 150 字\n"
                )
        
        # 呼叫 TalkToEBM
        print(f"正在為特徵 {feature_name} 生成 AI 解釋...")
        
        description = describe_graph(
            llm,
            ml_model.model,
            feature_index=feature_idx,
            num_sentences=1,
            max_chars=30,
            style="patient",
            temperature=0.7,
            custom_prompt=custom_prompt
        )
        
        print(f"✅ AI 解釋生成完成")
        
        return JsonResponse({
            'success': True,
            'explanation': description,
            'feature': feature_name,
            'patient_value': float(patient_value) if patient_value is not None else None
        })
        
    except Exception as e:
        import traceback
        print(f"❌ AI 解釋失敗: {e}")
        print(traceback.format_exc())
        return JsonResponse({
            'success': False,
            'error': str(e)
        })