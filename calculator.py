import tkinter as tk
from tkinter import ttk, messagebox

class CalculatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("簡易計算機")
        self.root.geometry("400x500")
        self.root.configure(bg="#f0f0f0")
        
        # 創建主框架
        main_frame = ttk.Frame(root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 輸入區域
        input_frame = ttk.LabelFrame(main_frame, text="輸入數值", padding=10)
        input_frame.pack(fill=tk.X, pady=10)
        
        # 第一個數值
        ttk.Label(input_frame, text="第一個數值:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num1_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.num1_var, width=20).grid(row=0, column=1, sticky=tk.W, pady=5)
        
        # 第二個數值
        ttk.Label(input_frame, text="第二個數值:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.num2_var = tk.StringVar()
        ttk.Entry(input_frame, textvariable=self.num2_var, width=20).grid(row=1, column=1, sticky=tk.W, pady=5)
        
        # 運算方式選擇
        ttk.Label(input_frame, text="運算方式:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.operation_var = tk.StringVar(value="加法")
        operation_combo = ttk.Combobox(input_frame, textvariable=self.operation_var, 
                                      values=["加法", "減法", "乘法", "除法", "次方"], 
                                      state="readonly", width=17)
        operation_combo.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # 結果區域
        result_frame = ttk.LabelFrame(main_frame, text="計算結果", padding=10)
        result_frame.pack(fill=tk.X, pady=10)
        
        self.result_var = tk.StringVar(value="")
        result_label = ttk.Label(result_frame, textvariable=self.result_var, font=("Arial", 14))
        result_label.pack(fill=tk.X, pady=10)
        
        # 按鈕區域
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        # 計算按鈕
        calculate_btn = ttk.Button(button_frame, text="計算結果", command=self.calculate)
        calculate_btn.pack(side=tk.LEFT, padx=5)
        
        # 清除按鈕
        clear_btn = ttk.Button(button_frame, text="清除輸入", command=self.clear_inputs)
        clear_btn.pack(side=tk.LEFT, padx=5)
        
        # 歷史記錄區域
        history_frame = ttk.LabelFrame(main_frame, text="計算歷史", padding=10)
        history_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.history_text = tk.Text(history_frame, height=10, width=40)
        self.history_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 歷史記錄滾動條
        scroll = ttk.Scrollbar(self.history_text, command=self.history_text.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.history_text.config(yscrollcommand=scroll.set)
        
        # 初始化空的歷史記錄
        self.history = []
    
    def calculate(self):
        """執行計算並顯示結果"""
        try:
            # 獲取輸入值
            num1 = float(self.num1_var.get())
            num2 = float(self.num2_var.get())
            operation = self.operation_var.get()
            
            # 執行計算
            result = 0
            if operation == "加法":
                result = num1 + num2
                op_symbol = "+"
            elif operation == "減法":
                result = num1 - num2
                op_symbol = "-"
            elif operation == "乘法":
                result = num1 * num2
                op_symbol = "×"
            elif operation == "除法":
                if num2 == 0:
                    raise ZeroDivisionError("除數不能為零")
                result = num1 / num2
                op_symbol = "÷"
            elif operation == "次方":
                result = num1 ** num2
                op_symbol = "^"
            
            # 格式化結果
            if result.is_integer():
                formatted_result = int(result)
            else:
                formatted_result = result
            
            # 更新結果顯示
            self.result_var.set(f"{formatted_result}")
            
            # 添加到歷史記錄
            history_entry = f"{num1} {op_symbol} {num2} = {formatted_result}"
            self.history.append(history_entry)
            self.update_history()
            
        except ValueError:
            messagebox.showerror("輸入錯誤", "請輸入有效的數字")
        except ZeroDivisionError:
            messagebox.showerror("計算錯誤", "除數不能為零")
        except Exception as e:
            messagebox.showerror("錯誤", f"計算時發生錯誤: {str(e)}")
    
    def clear_inputs(self):
        """清除輸入欄位和結果"""
        self.num1_var.set("")
        self.num2_var.set("")
        self.result_var.set("")
    
    def update_history(self):
        """更新歷史記錄顯示"""
        self.history_text.delete(1.0, tk.END)
        for i, entry in enumerate(reversed(self.history), 1):
            self.history_text.insert(tk.END, f"{i}. {entry}\n")

def main():
    root = tk.Tk()
    app = CalculatorApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 