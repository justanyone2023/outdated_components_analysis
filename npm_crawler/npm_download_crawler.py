import asyncio
import aiohttp
from clickhouse_driver import Client
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 定义 ClickHouse 客户端对象
clickhouse_client = None

def initialize_clickhouse_client():
    global clickhouse_client
    try:
        clickhouse_client = Client(host='', port=3306, user='', password='', database='')
    except Exception as e:
        print(f"Failed to initialize ClickHouse client: {e}")

async def fetch_download_info(package, session):
    url = f'https://api.npmjs.org/downloads/point/last-month/{package}'
    try:
        async with session.get(url, headers=headers, proxy=proxy) as response:
            if response.status == 200:
                info = await response.json()
                return info
            else:
                print(f"Failed to fetch download info for package: {package}. Retrying...")
    except aiohttp.ClientError as e:
        print(f"Request failed: {e}")
    return None

async def process_packages(package_names):
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_download_info(package, session) for package in package_names]
        return await asyncio.gather(*tasks)

def insert_download_info_to_clickhouse(download_info):
    initialize_clickhouse_client()  # Ensure ClickHouse client is initialized
    global clickhouse_client
    query = 'INSERT INTO npm_package_v1 (package, start, end, download_count) VALUES'
    for row in download_info:
        if row is not None:
            query += f"('{row['package']}', '{row['start']}', '{row['end']}', {row['downloads']}),"
    query = query.rstrip(',')  # 去除最后一个逗号
    clickhouse_client.execute(query)

async def main_async(package_names):
    chunk_size = 100  # 每次处理的包名数量
    total_packages = len(package_names)
    with tqdm(total=total_packages, desc='Processing packages') as pbar:
        for i in range(0, total_packages, chunk_size):
            chunk = package_names[i:i+chunk_size]
            download_info = await process_packages(chunk)
            insert_download_info_to_clickhouse(download_info)
            pbar.update(len(chunk))

def run_async_in_thread(package_names):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main_async(package_names))
    finally:
        loop.close()

def fetch_package_names_from_clickhouse():
    initialize_clickhouse_client()  # Ensure ClickHouse client is initialized
    global clickhouse_client
    query = """
    SELECT DISTINCT package
    FROM temp_package
    WHERE package NOT IN (
        SELECT DISTINCT package
        FROM npm_package_v1
    ) order by package;
    """
    result = clickhouse_client.execute(query)
    return [row[0] for row in result]

if __name__ == '__main__':
    package_names = fetch_package_names_from_clickhouse()

    num_threads = 8
    chunk_size = len(package_names) // num_threads
    packages_chunks = [package_names[i:i+chunk_size] for i in range(0, len(package_names), chunk_size)]

    # 使用线程池并发地执行异步任务
    with ThreadPoolExecutor() as executor:
        executor.map(run_async_in_thread, packages_chunks)
