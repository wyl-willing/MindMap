# MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models
This is the official codebase of the **MindMap** :snowflake: framework for eliciting the graph-of-thoughts reasoning capability in LLMs, proposed in [MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models](https://arxiv.org/pdf/2308.09729.pdf).

# Overview
We present **MindMap**, a plug-and-play prompting approach, which enables LLMs to comprehend graphical inputs to build their own mind map that supports evidence-grounded generation. Here is an overview of our model architecture:
![https://github.com/willing510/MindMap/blob/main/fig/mind%20map.png](https://github.com/willing510/MindMap/blob/main/fig/mind%20map.png)

# Run MindMap
As the chatdoctor5k dataset for example. First, you need to create a Blank Sandbox on [https://sandbox.neo4j.com/](https://sandbox.neo4j.com/), click "connect via drivers", find your url and user password. Then replace the following parts in MindMap.py:
```
uri = "Your_url"
username = "Your_user"     
password = "Your_password"
```
Note that the data of CMCKG is too large, and it will take about two days to wait. We recommend clicking "extend your project" in neo4j sandbox. But don't worry, the EMCKG used by chatdoctor5k will be ready to build on your facility in no time.
Then, don't forget to replace your openai_key in MindMap.py.


```
python MindMap.py
```

If you find this paper interesting, please consider cite it through

```latex
@article{wen2023mindmap,
  title={MindMap: Knowledge Graph Prompting Sparks Graph of Thoughts in Large Language Models},
  author={Wen, Yilin and Wang, Zifeng and Sun, Jimeng},
  journal={arXiv preprint arXiv:2308.09729},
  year={2023}
}
```
