from serial import Serial
from pyubx2 import UBXMessage, SET

def config_ublox(port: str, rate_ms: int, baudrate: int):
    
    baudrate_arr = (9600, 19200, 38400, 57600, 115200, 230400)

    for baud in baudrate_arr:
        with Serial(port, baud, timeout=1) as stream:
            # Настраиваем частоту сообщений (мс)
            msg_rate = UBXMessage(
                "CFG",
                "CFG-RATE",
                SET,
                measRate=rate_ms, 
                navRate=5, 
                timeRef=1,
            )
            stream.write(msg_rate.serialize())  # updating...
        
            # Настраиваем скорость UART1
            uart1set = UBXMessage(
                "CFG",
                "CFG-PRT",
                SET,
                portID=1,
                enable=0,
                pol=0,
                pin=0,
                thres=0,
                charLen=3,
                parity=4,
                nStopBits=0,
                baudRate=baudrate,
                inUBX=1,
                inNMEA=1,
                outUBX=0,
                outNMEA=1,
                extendedTxTimeout=0,
            )
            stream.write(uart1set.serialize())  # updating...

    # Сохраняем конфигурацию в энергонезависимой памяти
    with Serial(port, baudrate, timeout=1) as stream:
        # send command CFG-CFG
        msg_cfg = UBXMessage(
            "CFG",
            "CFG-CFG",
            SET,
            saveMask=b"\x1f\x1f\x00\x00", 
            devBBR=1,
            devFlash=1,  
            devEEPROM=1,
        )
        stream.write(msg_cfg.serialize())


if __name__ == "__main__":

    rate_ms = 100
    baudrate = 115200
    config_ublox("/dev/ttyS0", rate_ms, baudrate)
        