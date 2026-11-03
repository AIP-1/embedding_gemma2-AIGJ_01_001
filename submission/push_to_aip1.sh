#!/usr/bin/env bash
# 과제 3개를 AIP-1 조직의 공개 제출 저장소로 올린다. 맥 터미널에서 실행한다 (Jetson 아님).
# 필요: gh CLI 로그인(gh auth status), AIP-1 에 저장소를 만들 권한.
# 대상 저장소가 없으면 공개로 만들고, 이미 커밋이 있으면 push 가 거부되어 덮어쓰지 않는다.
set -euo pipefail
BR=claude/project-thread-db2w2p
SRC_TS=https://github.com/AIP-1-3/jetson_AIGJ_01_001.git
SRC_EG=https://github.com/AIP-1-3/jetson_embeddinggemma2.git
ORG=AIP-1
ID=AIGJ_01_001
WORK="$(mktemp -d)"; echo "작업 폴더: $WORK"

ensure_repo() {  # $1 = 저장소 이름
  if gh repo view "$ORG/$1" >/dev/null 2>&1; then echo "있음: $ORG/$1"
  else gh repo create "$ORG/$1" --public >/dev/null && echo "만듦: $ORG/$1"; fi
}
push_main() {  # $1 = 로컬 폴더, $2 = 저장소 이름
  git -C "$1" push "https://github.com/$ORG/$2.git" HEAD:main
  echo "완료: https://github.com/$ORG/$2"
}

# 1) 시계열: 그대로
N=timeseries-inference-$ID
ensure_repo "$N"
git clone -q --branch "$BR" "$SRC_TS" "$WORK/ts"
push_main "$WORK/ts" "$N"

# 2) 카메라 응용 (8~10단계): 이 브랜치의 README 그대로, submission/ 만 뺌
N=practice_task_for_jetson_camera_applications-$ID
ensure_repo "$N"
git clone -q --branch "$BR" "$SRC_EG" "$WORK/cam"
git -C "$WORK/cam" rm -r -q submission
git -C "$WORK/cam" commit -q -m "제출용: submission 폴더 제거"
push_main "$WORK/cam" "$N"

# 3) 임베딩 (1~7단계): README 를 1~7단계용으로 바꿈
N=embedding_gemma2-$ID
ensure_repo "$N"
git clone -q --branch "$BR" "$SRC_EG" "$WORK/eg"
cp "$WORK/eg/submission/README-embedding_gemma2.md" "$WORK/eg/README.md"
git -C "$WORK/eg" rm -r -q submission
git -C "$WORK/eg" add README.md
git -C "$WORK/eg" commit -q -m "제출용 README: 1~7단계 범위로 정리"
push_main "$WORK/eg" "$N"
