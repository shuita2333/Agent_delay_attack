模型构建函数未完成，需要更改具体模型生成

```
def load_indiv_model(self, model_name)
def get_model_path_and_template(model_name)
```

AttackLM类中get_attack方法Get prompts(需要根据模型修改)  74行

batched_generate（）函数没写 86行





judges修改返回值强制为1，观察循环过程

```
    def process_output(self, raw_output):
        pattern = r'\[\[(\d+)\]\]'
        match = re.search(pattern, raw_output)
        output = int(match.group(1)) if match else None
        if output is None:
            print(f"Error in processing judge output: {raw_output}" )
            output = 1
        # return output
        return 1
```

