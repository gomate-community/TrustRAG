import json
import re
from typing import List

import jieba
import loguru

from trustrag.modules.document.utils import PROJECT_BASE


class LLMCitation:
    def __init__(self):
        self.stopwords = ["的"]

    def cut(self, para: str):

        # 定义结束符号列表
        # end_symbols = ['。', '！', '？', '…', '；', '\n'] # sent
        end_symbols = ['。', '！', '？', '…', '；', '\n']  # para

        # 定义引号对
        quote_pairs = {'"': '"', "'": "'", '「': '」', '『': '』'}

        sentences = []
        current_sentence = ''
        quote_stack = []

        for char in para:
            current_sentence += char

            # 处理引号
            if char in quote_pairs.keys():
                quote_stack.append(char)
            elif quote_stack and char in quote_pairs.values():
                if char == quote_pairs[quote_stack[-1]]:
                    quote_stack.pop()

            # 当遇到结束符号且不在引号内时，进行分句
            if char in end_symbols and not quote_stack:
                # 去除可能的空白符号
                # sentence = current_sentence.strip()
                sentence = current_sentence
                if sentence:
                    sentences.append(sentence)
                current_sentence = ''

        # 处理末尾可能剩余的文本
        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    def remove_stopwords(self, query: str):
        for word in self.stopwords:
            query = query.replace(word, " ")
        return query

    def highlight_common_substrings(self, response_content, select_content, min_length=6):
        """
        Find the best matching content between response_content and select_content
        and return the start and end positions in select_content.

        Args:
            response_content: The response text to compare against
            select_content: The selected text to find matching positions in
            min_length: Minimum length of common substring to consider

        Returns:
            List of [start, end] positions in select_content that best match with response_content
        """
        response_sentences = self.cut(response_content)
        select_sentences = self.cut(select_content)

        best_match_ratio = 0
        best_match_positions = []

        # If there are no sentences, return empty result
        if not select_sentences:
            return []

        # Compare each sentence in response with each sentence in select_content
        for response_sentence in response_sentences:
            for i, select_sentence in enumerate(select_sentences):
                ratio = self.cal_common_ration(response_sentence, select_sentence)

                # If this match is better than our current best match, update
                if ratio > best_match_ratio and len(select_sentence) >= min_length:
                    best_match_ratio = ratio

                    # Find the position of this sentence in the original select_content
                    try:
                        start_pos = select_content.index(select_sentence)
                        end_pos = start_pos + len(select_sentence) - 1
                        best_match_positions = [[start_pos, end_pos]]
                    except ValueError:
                        # Handle case where the exact sentence can't be found (rare edge case)
                        continue

        # If no good match found, try to match the whole content
        if not best_match_positions and len(select_content) >= min_length:
            whole_ratio = self.cal_common_ration(response_content, select_content)
            if whole_ratio > 0.5:  # Some minimum threshold
                best_match_positions = [[0, len(select_content) - 1]]
        return best_match_positions

    def cal_common_ration(self, response, evidence):
        """
        计算答案中的段落与匹配证据的相似度，或者重合度，直接居于共现词的比例
        """
        sentence_seg_cut = set(jieba.lcut(self.remove_stopwords(response)))
        sentence_seg_cut_length = len(sentence_seg_cut)
        evidence_seg_cut = set(jieba.lcut(self.remove_stopwords(evidence)))
        overlap = sentence_seg_cut.intersection(evidence_seg_cut)
        ratio = len(overlap) / sentence_seg_cut_length
        return ratio

    def extract_citations(self, response: str = None):
        """
        xxx[1]xxx[2],
        find all citation patterns like [number]
        """
        citation_pattern = r'\[(\d+)\]'
        citations = []

        for match in re.finditer(citation_pattern, response):
            start, end = match.span()
            index = int(match.group(1))
            citations.append({
                "position": start,
                "citation": match.group(0),
                "index": index
            })

        # Parse the content into the required format
        parsed_result = []
        last_position = 0

        for citation in citations:
            # Add text before the citation
            if citation["position"] > last_position:
                text_content = response[last_position:citation["position"]]
                if text_content:
                    parsed_result.append({
                        "content": text_content,
                        "type": "text"
                    })

            # Add the citation
            parsed_result.append({
                "content": citation["citation"],
                "type": "citation",
                "index": citation["index"]
            })

            last_position = citation["position"] + len(citation["citation"])

        # Add remaining text after the last citation
        if last_position < len(response):
            parsed_result.append({
                "content": response[last_position:],
                "type": "text"
            })

        return {
            "citations": citations,
            "parsed_result": parsed_result
        }

    def ground_response(
            self,
            question: str,
            response: str,
            evidences: List[str],
            selected_idx: List[int],
            markdown: bool = True,
            show_code=False,
            selected_docs=List[dict]
    ):
        """
        """
        # Create JSON object
        json_data = {
            "question": question,
            "response": response,
            "evidences": evidences,
            "selected_idx": selected_idx,
            "selected_docs": selected_docs
        }

        # Save to JSON file
        try:
            output_file = "/home/yanqiang/code/citation_match_llm_res.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        except Exception as e:
            print(json_data)
            output_file = "citation_match_llm_res.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                loguru.logger.info(json_data)
                json.dump(json_data, f, ensure_ascii=False, indent=4)
        loguru.logger.info(f"Parameters saved to {output_file}")
        citation_result = self.extract_citations(response=response)
        parsed_result = citation_result["parsed_result"]
        print(citation_result)

        quote_list = []

        for idx, citation_item in enumerate(parsed_result):
            # todo:判断citation_item的类型，是text还是citation，
            # 如果当前citation_item是text，判断下一个类型是否为是citation，如果为citation，那么best_idx等于下面：
            # best_idx=parsed_result[idx+1]["index"]
            if idx <= len(parsed_result) - 2:
                if citation_item["type"] == "text":
                    if parsed_result[idx + 1]["type"] == "citation":
                        best_idx = parsed_result[idx + 1]["index"]  # 这个是selected_idx的真实引号+1，例如38
                        best_idx = selected_idx.index((int(best_idx) - 1))  #

                        print(best_idx)
                        response_content = citation_item["content"]
                        select_content = selected_docs[best_idx]["content"]

                        highlighted_start_end = self.highlight_common_substrings(response_content, select_content)
                        group_item = {
                            "doc_id": selected_docs[best_idx]["doc_id"],
                            "chk_id": selected_docs[best_idx]["chk_id"],
                            "doc_source": selected_docs[best_idx]["newsinfo"]["source"],
                            "doc_date": selected_docs[best_idx]["newsinfo"]["date"],
                            "doc_title": selected_docs[best_idx]["newsinfo"]["title"],
                            # "chk_content": selected_docs[best_idx]['content'],
                            "chk_content": select_content,
                            "best_ratio": self.cal_common_ration(response_content, select_content),
                            "highlight": highlighted_start_end,
                        }

                        group_data = {
                            "doc_list": [group_item],
                            "chk_content": group_item["chk_content"],
                            "highlight": group_item["highlight"],
                        }
                        quote_list.append(group_data)

        response_result = ''.join([item["content"] for item in citation_result["parsed_result"]])
        data = {'result': response_result, 'quote_list': quote_list, 'summary': ''}

        # Save to JSON file
        json_data['result'] = response_result
        json_data['quote_list'] = quote_list
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=4)
        loguru.logger.info(f"Parameters saved to {output_file}")

        return data


if __name__ == '__main__':
    mc = LLMCitation()

    with open(f'{PROJECT_BASE}/modules/citation/citation_match_llm.json', 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    result = mc.ground_response(
        question=input_data["question"],
        response=input_data["response"],
        evidences=input_data["evidences"],
        selected_idx=input_data["selected_idx"],
        markdown=True,
        show_code=True,
        selected_docs=input_data["selected_docs"],
    )

    print(result)
