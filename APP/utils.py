import aiofiles
import os


async def save_upload_file(upload_file, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    async with aiofiles.open(destination, "wb") as out_file:
        content = await upload_file.read()
        await out_file.write(content)
