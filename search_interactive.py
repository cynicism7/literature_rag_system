"""
交互式搜索界面
可以连续输入多个查询
"""
import sys
from run_search import search

def main():
    print("="*80)
    print("RAG系统 - 交互式搜索")
    print("="*80)
    print("输入查询文本进行搜索，输入 'quit' 或 'exit' 退出")
    print("="*80)
    
    while True:
        try:
            query = input("\n请输入查询: ").strip()
            
            if not query:
                continue
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
            
            # 询问top_k
            top_k_input = input("返回结果数量 (默认10): ").strip()
            top_k = int(top_k_input) if top_k_input else 10
            
            print("\n")
            search(query, top_k=top_k)
            
        except KeyboardInterrupt:
            print("\n\n再见！")
            break
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

