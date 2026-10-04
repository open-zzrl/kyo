from kyo import Kyo


def main():
    print("Инициализация Kyo напрямую из Hugging Face Hub (open-zzrl/kyo)...")
    engine = Kyo.from_pretrained("open-zzrl/kyo", confidence_threshold=0.85)

    telemetry = {
        "event": "AuthenticationAudit",
        "target_endpoint": "/api/v2/tokens/rotate",
        "failed_attempts_last_60s": 52,
        "ip_reputation_score": 0.89,
        "geo_velocity_anomaly": True,
    }

    instruction = "Assess the threat severity of this authentication telemetry log."
    options = {
        "Benign": "Routine or expected read-only operations.",
        "Low": "Isolated failed password attempt or transient network glitch.",
        "Moderate": "Repeated anomalies from a single endpoint, potential probe.",
        "Critical": "Active brute-force or high-velocity credential stuffing attack.",
    }

    print("Выполнение инференса...")
    result = engine.decide(
        context=telemetry, instruction=instruction, options=options
    )

    print("-" * 60)
    print(f"Результат          : {result}")
    print(f"Выбранное действие : {result.decision}")
    print(f"Уверенность (Conf) : {result.confidence * 100:.2f}%")
    print(f"Задержка (Latency) : {result.latency_ms:.2f} ms")
    print(f"Нужен Fallback LLM: {result.fallback_to_llm}")
    print("Распределение скоров:")
    for opt, prob in result.scores.items():
        print(f"  - {opt:<10}: {prob * 100:>6.2f}%")
    print("-" * 60)


if __name__ == "__main__":
    main()
