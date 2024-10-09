# langchain-pangu

## 使用

```shell
pip install langchain-pangu
```

## 升级到 0.3

上游 langchain、langgraph、pedantic 等依赖都进行了大版本的升级，为跟随上游更新，本项目也更新到 0.3 版本

注意 0.3 与之前的版本不兼容，使用 0.3 版本需要将所有依赖都进行升级

## dev

```shell
# 安装依赖
pip install .
# 打包
python3 -m build
# 上传
twine upload dist/*
```

## 第三方代码版权声明

代码中使用的 [`langchain_pangu/pangukitsappdev`](langchain_pangu/pangukitsappdev) 是第三方库 `pangu-kits-app-dev-py` 的内容，因为其依赖冲突无法解决，将其引入本项目进行处理。

```python
#  Copyright (c) Huawei Technologies Co., Ltd. 2023-2023. All rights reserved.
```
