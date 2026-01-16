"""
交互式智能问答界面
支持连续提问，自动提取关键词并生成答案
"""
import sys
from run_qa import qa_search

def main():
    print("="*80)
    print("RAG系统 - 智能问答")
    print("="*80)
    print("输入问题，系统会自动提取关键词、检索相关内容并生成答案")
    print("输入 'quit' 或 'exit' 退出")
    print("="*80)
    
    while True:
        try:
            query = input("\n请输入您的问题: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            # 询问参数
            top_k_input = input("检索结果数量 (默认10): ").strip()
            top_k = int(top_k_input) if top_k_input else 10
            
            answer_n_input = input("用于生成答案的chunk数量 (默认3): ").strip()
            answer_top_n = int(answer_n_input) if answer_n_input else 3
            
            print("\n")
            qa_search(query, top_k=top_k, answer_top_n=answer_top_n)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

