#!/bin/bash

# カレントディレクトリ内のすべてのシンボリックリンクを探索
for symlink in $(find . -maxdepth 1 -type l); do
  # シンボリックリンクが指している実体のパスを取得
  target=$(readlink -f "$symlink")
  
  # 実体がディレクトリであるか確認
  if [ -d "$target" ]; then
    # シンボリックリンクを削除
    rm "$symlink"
    
    # 実体のディレクトリをシンボリックリンクの場所にコピー
    cp -r "$target" "$symlink"
  fi
done

