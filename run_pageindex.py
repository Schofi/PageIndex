import argparse
from pageindex import *
import os

# 默认配置
DEFAULT_MODEL = "deepseek-chat"
DEFAULT_API_BASE = "https://api.deepseek.com"
DEFAULT_API_KEY = "your-api-key-here"

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process PDF document and generate structure')
    parser.add_argument('--pdf_path', type=str, help='Path to the PDF file')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL, help='Model to use')
    parser.add_argument('--api_base', type=str, default=DEFAULT_API_BASE, 
                      help='API base URL (default: https://api.deepseek.com)')
    parser.add_argument('--api_key', type=str, default=DEFAULT_API_KEY,
                      help='API key for authentication')
    parser.add_argument('--toc-check-pages', type=int, default=20, 
                      help='Number of pages to check for table of contents')
    parser.add_argument('--max-pages-per-node', type=int, default=10,
                      help='Maximum number of pages per node')
    parser.add_argument('--max-tokens-per-node', type=int, default=20000,
                      help='Maximum number of tokens per node')
    parser.add_argument('--if-add-node-id', type=str, default='yes',
                      help='Whether to add node id to the node')
    parser.add_argument('--if-add-node-summary', type=str, default='no',
                      help='Whether to add summary to the node')
    parser.add_argument('--if-add-doc-description', type=str, default='yes',
                      help='Whether to add doc description to the doc')
    args = parser.parse_args()

    # 设置环境变量
    os.environ["CHATGPT_API_KEY"] = args.api_key
    os.environ["CHATGPT_BASE_URL"] = args.api_base
        
    # Configure options
    opt = config(
        model=args.model,
        toc_check_page_num=args.toc_check_pages,
        max_page_num_each_node=args.max_pages_per_node,
        max_token_num_each_node=args.max_tokens_per_node,
        if_add_node_id=args.if_add_node_id,
        if_add_node_summary=args.if_add_node_summary,
        if_add_doc_description=args.if_add_doc_description
    )

    # Process the PDF
    toc_with_page_number = page_index_main(args.pdf_path, opt)
    print('Parsing done, saving to file...')
    
    # Save results
    pdf_name = os.path.splitext(os.path.basename(args.pdf_path))[0]    
    os.makedirs('./results', exist_ok=True)
    
    with open(f'./results/{pdf_name}_structure.json', 'w', encoding='utf-8') as f:
        json.dump(toc_with_page_number, f, indent=2, ensure_ascii=False)