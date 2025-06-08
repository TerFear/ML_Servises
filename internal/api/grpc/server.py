from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import time
import gen.py.ml_pb2 as ml_pb2
import gen.py.ml_pb2_grpc as ml_pb2_grpc


class MLModelServicer(ml_pb2_grpc.MLModelServicer):
    def __init__(self):
        # Загружаем модель один раз при старте
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Embedding-0.6B")

    def GenerateUnary(self, request, context):
        print(f"[Unary] Получен запрос: {request.data}, тип: {request.type}")
        inputs = self.tokenizer(request.data, return_tensors="pt")

        generate_ids = self.model.generate(inputs.input_ids, max_length=30)
        res = self.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return ml_pb2.ModelResponse(
            text=res,
            model_name="gpt2"
        )

    def GenerateStream(self, request, context):
        print(f"[Stream] Получен запрос: {request.data}, тип: {request.type}")
        # Для простоты разобьём текст по словам
        result = self.generator("Привет! Расскажи анекдот", max_length=80, do_sample=True)[0]["generated_text"]
        tokens = result.split()
        for token in tokens:
            yield ml_pb2.TokenResponse(token=token + " ")
            time.sleep(0.2)  # эмуляция стриминга
