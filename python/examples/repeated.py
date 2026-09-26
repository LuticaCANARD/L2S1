"""python python/examples/repeated.py /path/to/model.gguf --binary /path/to/l2s1"""
import argparse
import asyncio

from pydantic import BaseModel, ConfigDict

from l2s1 import BinaryKind, Decision, L2S1, LoadOptions


class Temperature(BaseModel):
    model_config = ConfigDict(strict=True)
    temperature_c: float


async def main(model: str, binary: str | None) -> None:
    async with await L2S1.load(LoadOptions(model=model, binary_path=binary, execution_mode="parallel")) as engine:
        plan = engine.prepare([Decision(
            id="cold", instruction="Is temperature_c below 10?",
            kind=BinaryKind(false_label="At least 10.", true_label="Below 10."),
        )], state_type=Temperature)
        first = await plan.decide(Temperature(temperature_c=6))
        print(first.model_dump_json(exclude_unset=True))
        for response in await plan.decide_batch([Temperature(temperature_c=15), Temperature(temperature_c=2)]):
            print(response.model_dump_json(exclude_unset=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--binary")
    args = parser.parse_args()
    asyncio.run(main(args.model, args.binary))
