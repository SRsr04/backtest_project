def breakeven_info(entry_price: float, volume: float, side: str, fee_percent: float = 0.0325):
    """
    Розраховує потенційну комісію і ціну беззбитковості (для стопу)
    
    :param entry_price: ціна входу
    :param volume: обʼєм угоди
    :param side: "long" або "short"
    :param fee_percent: комісія за сторону в %
    """
    notional = entry_price * volume
    total_fee = notional * (fee_percent / 100) * 2  # подвійна комісія (вхід + вихід)

    if side.lower() == "long":
        breakeven_price = (notional + total_fee) / volume
    elif side.lower() == "short":
        breakeven_price = (notional - total_fee) / volume
    else:
        print("❌ Сторона угоди має бути 'long' або 'short'")
        return

    print(f"📌 Потенційна комісія: {total_fee:.2f} USDT")
    print(f"🛡 Ціна беззбитковості (S/L): {breakeven_price:.2f} USDT")


    
# print(breakeven_price(2388.51, 0.39, 1.67, 'short'))



print(breakeven_info(116245.0, 0.22, 'long'))