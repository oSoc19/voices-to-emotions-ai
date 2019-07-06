import hashlib


def filehash(file_path, chunk_size=65536):
    sha1 = hashlib.sha1()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(chunk_size)

            if not data:
                break

            sha1.update(data)

    return sha1.hexdigest()
