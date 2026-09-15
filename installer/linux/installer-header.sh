#!/bin/sh
set -eu

APP_NAME="OGDD"
INSTALL_ROOT="${XDG_DATA_HOME:-${HOME}/.local/share}/ogdd"
APP_DIR="${INSTALL_ROOT}/OGDD"
DESKTOP_DIR="${XDG_DATA_HOME:-${HOME}/.local/share}/applications"
DESKTOP_FILE="${DESKTOP_DIR}/org.ogdd.OGDD.desktop"

archive_line=$(awk '/^__OGDD_ARCHIVE_BELOW__$/{print NR + 1; exit}' "$0")
if [ -z "${archive_line}" ]; then
    echo "No se encontró el contenido del instalador de ${APP_NAME}." >&2
    exit 1
fi

staging_dir=$(mktemp -d "${TMPDIR:-/tmp}/ogdd-install.XXXXXX")
trap 'rm -rf "${staging_dir}"' EXIT HUP INT TERM

tail -n +"${archive_line}" "$0" | tar -xz -C "${staging_dir}"
test -x "${staging_dir}/OGDD/OGDD"

mkdir -p "${INSTALL_ROOT}" "${DESKTOP_DIR}"
if [ -d "${APP_DIR}" ]; then
    backup_dir="${INSTALL_ROOT}/OGDD.previous"
    rm -rf "${backup_dir}"
    mv "${APP_DIR}" "${backup_dir}"
fi
mv "${staging_dir}/OGDD" "${APP_DIR}"

escaped_executable=$(printf '%s' "${APP_DIR}/OGDD" | sed 's/[&|]/\\&/g')
sed "s|@OGDD_EXECUTABLE@|${escaped_executable}|g" \
    "${APP_DIR}/ogdd.desktop" > "${DESKTOP_FILE}"
chmod 755 "${APP_DIR}/OGDD"
chmod 644 "${DESKTOP_FILE}"

cat > "${INSTALL_ROOT}/uninstall-ogdd.sh" <<EOF
#!/bin/sh
set -eu
rm -rf "${APP_DIR}"
rm -f "${DESKTOP_FILE}"
rm -f "${INSTALL_ROOT}/uninstall-ogdd.sh"
echo "OGDD fue desinstalado."
EOF
chmod 755 "${INSTALL_ROOT}/uninstall-ogdd.sh"

rm -rf "${INSTALL_ROOT}/OGDD.previous"
command -v update-desktop-database >/dev/null 2>&1 && \
    update-desktop-database "${DESKTOP_DIR}" >/dev/null 2>&1 || true

echo "OGDD quedó instalado para el usuario actual."
echo "Puede iniciarlo desde el menú de aplicaciones o con: ${APP_DIR}/OGDD"
exit 0

__OGDD_ARCHIVE_BELOW__
