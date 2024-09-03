#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.

from typing import Optional, Any, Dict, Union, List

from langchain.callbacks.manager import CallbackManagerForChainRun
from langchain.chains import RetrievalQAWithSourcesChain


class RetrievalQAWithAllSourcesChain(RetrievalQAWithSourcesChain):
    """
    不依赖LLM的输出获取答案的来源，而是根据查询出来的doc数据直接获取source信息
    """

    @property
    def _chain_type(self) -> str:
        return "RetrievalQAWithAllSourcesChain"

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Union[str, List]]:
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = self._get_docs(inputs)
        answer = self.combine_documents_chain.run(
            input_documents=docs, callbacks=_run_manager.get_child(), **inputs
        )

        result: Dict[str, Any] = {
            self.answer_key: answer,
            self.sources_answer_key: [d.metadata for d in docs],
        }
        if self.return_source_documents:
            result["source_documents"] = docs
        return result
