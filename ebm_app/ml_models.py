import joblib
import pandas as pd
from plotly.offline import plot
import numpy as np
from scipy.ndimage import gaussian_filter1d

class MLInterpretModel:
    def __init__(self, model_path, data_path, feature_cols, target_col):
        self.model = joblib.load(model_path)
        self.data = pd.read_csv(data_path, encoding='utf-8-sig')
        self.feature_cols = feature_cols
        self.target_col = target_col
        
        # 統一將 ID 轉為字串
        self.data['ID'] = self.data['ID'].astype(str)
        
        # 儲存當前病人 ID 與特徵值
        self.current_patient_id = None
        self.current_patient_values = {}
    
    # ✅ 新增：計算局部梯度（平滑後求斜率）
    def calculate_local_gradient(self, x_values, y_values, patient_value, sigma=2):
        """
        計算病人值附近的局部梯度
        
        參數:
            x_values: 特徵值陣列
            y_values: 對應的預測貢獻度
            patient_value: 病人的特徵值
            sigma: 高斯平滑參數（越大越平滑）
        
        返回:
            gradient: 局部梯度值
            recommendation: 建議方向 ('decrease', 'increase', 'maintain')
            y_smooth: 平滑後的 y 值
            x_sorted: 排序後的 x 值
        """
        try:
            # 確保資料為 numpy array
            x_vals = np.array(x_values)
            y_vals = np.array(y_values)
            
            # 排序（確保 x 遞增）
            sorted_indices = np.argsort(x_vals)
            x_sorted = x_vals[sorted_indices]
            y_sorted = y_vals[sorted_indices]
            
            # 平滑化 y 值（避免鋸齒）
            y_smooth = gaussian_filter1d(y_sorted, sigma=sigma)
            
            # 找到病人值在 x 軸上的位置
            idx = np.searchsorted(x_sorted, patient_value)
            
            # 處理邊界情況
            if idx == 0:
                idx = 1
            elif idx >= len(x_sorted):
                idx = len(x_sorted) - 1
            
            # 使用中央差分法計算梯度（更準確）
            if idx > 0 and idx < len(x_sorted) - 1:
                # 中央差分：(f(x+h) - f(x-h)) / (2h)
                dx = (x_sorted[idx + 1] - x_sorted[idx - 1])
                dy = (y_smooth[idx + 1] - y_smooth[idx - 1])
                gradient = dy / dx if dx != 0 else 0
            elif idx == 0:
                # 前向差分
                dx = x_sorted[idx + 1] - x_sorted[idx]
                dy = y_smooth[idx + 1] - y_smooth[idx]
                gradient = dy / dx if dx != 0 else 0
            else:
                # 後向差分
                dx = x_sorted[idx] - x_sorted[idx - 1]
                dy = y_smooth[idx] - y_smooth[idx - 1]
                gradient = dy / dx if dx != 0 else 0
            
            # 根據梯度給出建議
            threshold = 0.001  # 設定一個閾值，避免微小梯度誤判
            if gradient > threshold:
                recommendation = 'decrease'  # 斜率 > 0，往右風險更高，建議降低
            elif gradient < -threshold:
                recommendation = 'increase'  # 斜率 < 0，往右風險更低，建議提高
            else:
                recommendation = 'maintain'  # 斜率接近 0，維持現狀
            
            return gradient, recommendation, y_smooth, x_sorted
            
        except Exception as e:
            print(f"計算梯度時發生錯誤: {e}")
            return 0, 'maintain', y_values, x_values
    
    # 尋找最佳目標值
    def find_optimal_target(self, x_sorted, y_smooth, patient_value, recommendation, search_range=0.3):
        """
        根據建議方向，在合理範圍內尋找風險最低的目標值
        
        參數:
            x_sorted: 排序後的特徵值
            y_smooth: 平滑後的貢獻度
            patient_value: 病人當前值
            recommendation: 建議方向 ('decrease', 'increase', 'maintain')
            search_range: 搜尋範圍（佔全範圍的比例，預設 30%）
        
        返回:
            target_value: 建議的目標值
            target_risk: 目標值的風險貢獻度
            risk_reduction: 預期風險降幅
        """
        try:
            # 如果建議維持現狀，直接返回
            if recommendation == 'maintain':
                current_idx = np.searchsorted(x_sorted, patient_value)
                if current_idx >= len(y_smooth):
                    current_idx = len(y_smooth) - 1
                return patient_value, y_smooth[current_idx], 0.0
            
            # 取得病人當前值的風險
            current_idx = np.searchsorted(x_sorted, patient_value)
            if current_idx >= len(y_smooth):
                current_idx = len(y_smooth) - 1
            current_risk = y_smooth[current_idx]
            
            # 計算搜尋範圍
            x_range = np.max(x_sorted) - np.min(x_sorted)
            search_distance = x_range * search_range
            
            # 根據建議方向設定搜尋區間
            if recommendation == 'decrease':
                # 往左搜尋（降低值）
                search_min = max(np.min(x_sorted), patient_value - search_distance)
                search_max = patient_value
                mask = (x_sorted >= search_min) & (x_sorted <= search_max)
            else:  # 'increase'
                # 往右搜尋（提高值）
                search_min = patient_value
                search_max = min(np.max(x_sorted), patient_value + search_distance)
                mask = (x_sorted >= search_min) & (x_sorted <= search_max)
            
            # 在搜尋範圍內找最小風險
            search_x = x_sorted[mask]
            search_y = y_smooth[mask]
            
            if len(search_y) == 0:
                # 沒有可搜尋的點，返回當前值
                return patient_value, current_risk, 0.0
            
            # 找到最小風險的索引
            min_idx = np.argmin(search_y)
            target_value = search_x[min_idx]
            target_risk = search_y[min_idx]
            
            # 計算風險降幅
            risk_reduction = current_risk - target_risk
            
            return target_value, target_risk, risk_reduction
            
        except Exception as e:
            print(f"尋找最佳目標值時發生錯誤: {e}")
            current_idx = np.searchsorted(x_sorted, patient_value)
            if current_idx >= len(y_smooth):
                current_idx = len(y_smooth) - 1
            return patient_value, y_smooth[current_idx], 0.0
    
    # ----------------------------
    # 全域解釋（加入梯度分析）
    # ----------------------------
    def get_global_explanation_html(self, feature=None, density_window=False, 
                                lower_percentile=2.5, upper_percentile=97.5):
        try:
            import plotly.graph_objects as go
            import numpy as np
            ebm_global = self.model.explain_global()
            
            if feature and feature in self.feature_cols:
                fig = ebm_global.visualize(self.feature_cols.index(feature))
                
                # 🆕 密度視窗過濾邏輯
                if density_window and len(fig.data) > 0:
                    for trace in fig.data:
                        if hasattr(trace, 'x') and hasattr(trace, 'y'):
                            x_vals = np.array(trace.x)
                            y_vals = np.array(trace.y)
                            
                            # 計算百分位數邊界
                            x_lower = np.percentile(x_vals, lower_percentile)
                            x_upper = np.percentile(x_vals, upper_percentile)
                            
                            # 過濾資料
                            mask = (x_vals >= x_lower) & (x_vals <= x_upper)
                            trace.x = x_vals[mask]
                            trace.y = y_vals[mask]

                # ✅ 如果有當前病人 → 進行梯度分析並標示
                if self.current_patient_id and feature in self.current_patient_values:
                    patient_value = self.current_patient_values[feature]
                    
                    # 取得原始曲線資料
                    if len(fig.data) > 0 and hasattr(fig.data[0], 'x'):
                        x_vals = np.array(fig.data[0].x)
                        y_vals = np.array(fig.data[0].y)
                        
                        # 計算局部梯度
                        gradient, recommendation, y_smooth, x_sorted = self.calculate_local_gradient(
                            x_vals, y_vals, patient_value, sigma=3
                        )
                        target_value, target_risk, risk_reduction = self.find_optimal_target(
                            x_sorted, y_smooth, patient_value, recommendation
                        )
                        
                        
                        # 準備建議文字和顏色
                        if recommendation == 'decrease':
                            arrow = "◀======== 建議降低"
                            color = "#FF6B6B"
                            annotation_pos = "top right"   # ⟸ 向左 → 文字放右
                            if risk_reduction > 0:
                                suggestion = f"降低至 {target_value:.2f} 可降低風險 {risk_reduction:.4f}"
                            else:
                                suggestion = f"降低此特徵可能降低風險"
                        elif recommendation == 'increase':
                            arrow = "========▶ 建議提高"
                            color = "#FF6B6B"
                            annotation_pos = "top left"    # ⟹ 向右 → 文字放左
                            if risk_reduction > 0:
                                suggestion = f"提高至 {target_value:.2f} 可降低風險 {risk_reduction:.4f}"
                            else:
                                suggestion = f"提高此特徵可能降低風險"
                        else:
                            arrow = "↔️ 維持現狀"
                            color = "#FFA500"
                            annotation_pos = "top left"
                            suggestion = f"此特徵值處於平穩區域，無需調整"
                        
                        # 添加病人值標記線
                        fig.add_vline(
                            x=patient_value,
                            line_dash="dash",
                            line_color=color,
                            line_width=3,
                            annotation_text=f"病人值: {patient_value:.2f}<br>{arrow}",
                            annotation_position=annotation_pos,
                            annotation_font=dict(size=14, color=color)
                        )

                        # 標記病人值點
                        # 找到最接近的 y 值
                        idx = np.searchsorted(np.sort(x_vals), patient_value)
                        if idx >= len(y_smooth):
                            idx = len(y_smooth) - 1
                        patient_y = y_smooth[idx]
                        
                        fig.add_trace(go.Scatter(
                            x=[patient_value],
                            y=[patient_y],
                            mode='markers',
                            marker=dict(size=15, color=color, symbol='diamond', 
                                      line=dict(width=2, color='white')),
                            name='當前病人',
                            showlegend=True,
                            hovertemplate=(
                                f'<b>當前病人</b><br>'
                                f'特徵值: {patient_value:.2f}<br>'
                                f'貢獻度: {patient_y:.4f}<br>'
                                f'局部梯度: {gradient:.4f}<br>'
                                f'<b>{suggestion}</b><extra></extra>'
                            )
                        ))
                        # ✅ 顯示目標點
                        fig.add_vline(
                            x=target_value,
                            line_dash="dot",
                            line_color="#000000",
                            line_width=2,
                            annotation_text=f"目標值: {target_value:.2f}",
                            annotation_position="bottom right",
                            annotation_font=dict(size=14, color="#44C767")
                        )

                        # ✅ 在目標點加上 marker
                        fig.add_trace(go.Scatter(
                            x=[target_value],
                            y=[target_risk],
                            mode='markers+text',
                            marker=dict(size=16, symbol='star', line=dict(width=2, color='white')),
                            text=[f"⬇風險 {risk_reduction:.4f}"],
                            textposition='bottom center',
                            name='建議目標',
                            showlegend=True,
                            hovertemplate=(
                                f'<b>建議目標</b><br>'
                                f'特徵值: {target_value:.2f}<br>'
                                f'預測風險: {target_risk:.4f}<br>'
                                f'風險下降: {risk_reduction:.4f}<extra></extra>'
                            )
                        ))

                        
                        # # 可選：顯示切線（視覺化梯度方向）
                        # # 計算切線的起點和終點
                        # x_range = np.max(x_vals) - np.min(x_vals)
                        # tangent_length = x_range * 0.1  # 切線長度為範圍的 10%
                        
                        # x_tangent = [patient_value - tangent_length, patient_value + tangent_length]
                        # y_tangent = [patient_y - gradient * tangent_length, 
                        #            patient_y + gradient * tangent_length]
                        
                        # fig.add_trace(go.Scatter(
                        #     x=x_tangent,
                        #     y=y_tangent,
                        #     mode='lines',
                        #     line=dict(color=color, width=2, dash='dot'),
                        #     name='局部趨勢',
                        #     showlegend=True,
                        #     hovertemplate=f'局部梯度: {gradient:.4f}<extra></extra>'
                        # ))
                    
            else:
                fig = ebm_global.visualize()
            
            # 統一修改全域 bar 顏色
            fig.update_traces(
                marker_color="#E0B859",
                marker_line_color="white",
                marker_line_width=1.5,
                selector=dict(type='bar')
            )

            # 格式調整
            if hasattr(fig, 'update_xaxes'):
                fig.update_xaxes(
                    tickangle=-45,
                    tickmode='auto',
                    nticks=10,
                    tickfont=dict(size=10)
                )
            
            fig.update_yaxes(autorange=True)
            fig.update_xaxes(autorange=True)

            fig.update_layout(
                autosize=True,
                margin=dict(l=50, r=50, t=50, b=80),
                yaxis=dict(
                    automargin=True,
                    fixedrange=False
                ),
                xaxis=dict(
                    automargin=True,
                    fixedrange=False
                ),
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor="rgba(255,255,255,0.8)"
                )
            )
            
            plot_div = plot(fig, output_type='div', include_plotlyjs='cdn', config={'responsive': True})
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        overflow: hidden;
                    }}
                    .plotly-graph-div {{
                        width: 100% !important;
                        height: 100vh !important;
                    }}
                </style>
            </head>
            <body>
                {plot_div}
                <script>
                    window.addEventListener('resize', function() {{
                        if (window.Plotly) {{
                            document.querySelectorAll('.plotly-graph-div').forEach(function(div) {{
                                Plotly.Plots.resize(div);
                            }});
                        }}
                    }});
                    window.addEventListener('load', function() {{
                        setTimeout(function() {{
                            if (window.Plotly) {{
                                document.querySelectorAll('.plotly-graph-div').forEach(function(div) {{
                                    Plotly.Plots.resize(div);
                                }});
                            }}
                        }}, 100);
                    }});
                </script>
            </body>
            </html>
            """
            return html
        
        except Exception as e:
            import traceback
            return f"""
            <!DOCTYPE html>
            <html>
            <body>
                <div style='text-align:center; padding:50px; color:red;'>
                    <h3>載入全域解釋時發生錯誤</h3>
                    <p>{str(e)}</p>
                    <pre style='text-align:left; font-size:10px;'>{traceback.format_exc()}</pre>
                </div>
            </body>
            </html>
            """

    # ----------------------------
    # 區域解釋
    # ----------------------------
    def get_local_explanation_html(self, patient_id, display_mode='all'):
        try:
            import plotly.graph_objects as go
            
            patient_id = str(patient_id)
            patient_df = self.data[self.data['ID'] == patient_id]
            
            if patient_df.empty:
                return """
                <!DOCTYPE html>
                <html>
                <body>
                    <div style='text-align:center; padding:50px; color:#f5576c;'>
                        <h3>⚠️ 找不到該病人資料</h3>
                        <p>請確認病人 ID 是否正確</p>
                    </div>
                </body>
                </html>
                """
            
            first_row = patient_df.iloc[[0]]
            X_first = first_row[self.feature_cols]
            y_first = first_row[self.target_col]
            
            self.current_patient_id = patient_id
            self.current_patient_values = X_first.iloc[0].to_dict()
            
            ebm_local = self.model.explain_local(X_first, y_first)
            local_data = ebm_local.data(0)
            
            feature_names = local_data['names']
            feature_scores = local_data['scores']
            feature_values = local_data['values']
            
            # 移除 intercept
            filtered_data = [
                {'name': n, 'score': s, 'value': v}
                for n, s, v in zip(feature_names, feature_scores, feature_values)
                if 'intercept' not in n.lower()
            ]
            
            # 根據顯示模式過濾資料
            if display_mode == 'positive':
                filtered_data = [d for d in filtered_data if d['score'] > 0]
                mode_text = "危險特徵（正貢獻）"
            elif display_mode == 'negative':
                filtered_data = [d for d in filtered_data if d['score'] < 0]
                mode_text = "安全特徵（負貢獻）"
            else:  # 'all'
                mode_text = "全部特徵"
            
            filtered_data.sort(key=lambda x: abs(x['score']), reverse=True)
            
            sorted_names = [d['name'] for d in filtered_data]
            sorted_scores = [d['score'] for d in filtered_data]
            sorted_values = [d['value'] for d in filtered_data]
            
            # 顏色根據模式調整
            if display_mode == 'positive':
                colors = ["#FF6B6B"] * len(sorted_scores)
            elif display_mode == 'negative':
                colors = ["#44C767"] * len(sorted_scores)
            else:
                colors = ["#FF6B6B" if s > 0 else "#44C767" for s in sorted_scores]
                
            labels = list(sorted_names)
            bar_texts = [f"值: {v}" for v in sorted_values]
            
            hover_texts = [
                f"<b>{n}</b><br>特徵值: {v}<br>貢獻度: {s:.4f}"
                for n, v, s in zip(sorted_names, sorted_values, sorted_scores)
            ]
            
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=labels,
                x=sorted_scores,
                orientation='h',
                marker=dict(color=colors),
                text=bar_texts,
                textposition='inside',
                textfont=dict(color='white', size=11, family='Arial'),
                insidetextanchor='middle',
                hovertext=hover_texts,
                hoverinfo='text',
                textangle=0
            ))
            
            fig.update_layout(
                title=f"病人 {patient_id} 的特徵貢獻度分析 - {mode_text}",
                xaxis_title="對預測的貢獻度",
                yaxis_title="特徵",
                height=max(600, len(sorted_names) * 40),
                showlegend=False,
                autosize=True,
                margin=dict(l=200, r=50, t=80, b=50),
                yaxis=dict(
                    tickfont=dict(size=11),
                    autorange="reversed"
                )
            )
            
            plot_div = plot(fig, output_type='div', include_plotlyjs='cdn', config={'responsive': True})
            
            html = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <style>
                    body {{
                        margin: 0;
                        padding: 0;
                        overflow-y: auto;
                        overflow-x: hidden;
                    }}
                    .plotly-graph-div {{
                        width: 100% !important;
                        height: auto !important;
                        min-height: 600px;
                    }}
                </style>
            </head>
            <body>
                {plot_div}
                <script>
                    window.addEventListener('resize', function() {{
                        if (window.Plotly) {{
                            document.querySelectorAll('.plotly-graph-div').forEach(function(div) {{
                                Plotly.Plots.resize(div);
                            }});
                        }}
                    }});
                    window.addEventListener('load', function() {{
                        setTimeout(function() {{
                            if (window.Plotly) {{
                                document.querySelectorAll('.plotly-graph-div').forEach(function(div) {{
                                    Plotly.Plots.resize(div);
                                }});
                            }}
                        }}, 100);
                    }});
                </script>
            </body>
            </html>
            """
            return html
        
        except Exception as e:
            import traceback
            return f"""
            <!DOCTYPE html>
            <html>
            <body>
                <div style='text-align:center; padding:50px; color:red;'>
                    <h3>載入區域解釋時發生錯誤</h3>
                    <p>{str(e)}</p>
                    <pre style='text-align:left; font-size:10px;'>{traceback.format_exc()}</pre>
                </div>
            </body>
            </html>
            """