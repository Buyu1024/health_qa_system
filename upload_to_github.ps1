# 🚀 一键上传到 GitHub 脚本
# 使用前请将 your-username 替换为你的 GitHub 用户名

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  医疗健康 RAG 系统 - GitHub 上传脚本" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 检查 Git 是否安装
try {
    $gitVersion = git --version
    Write-Host "✅ Git 已安装: $gitVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ 错误: 未检测到 Git，请先安装 Git" -ForegroundColor Red
    Write-Host "下载地址: https://git-scm.com/download/win" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "📋 步骤 1: 检查敏感文件状态..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 显示当前 Git 状态
git status

Write-Host ""
Write-Host "⚠️  请检查以上输出，确保 config.ini 没有被追踪" -ForegroundColor Yellow
Write-Host "   如果 config.ini 出现在 'Changes to be committed' 中，请按 Ctrl+C 停止" -ForegroundColor Yellow
Write-Host ""

$confirmation = Read-Host "确认继续上传？(y/n)"
if ($confirmation -ne 'y') {
    Write-Host " 已取消上传" -ForegroundColor Red
    exit 0
}

Write-Host ""
Write-Host "📦 步骤 2: 添加文件到 Git..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 添加所有文件
git add .

# 显示将要提交的文件
Write-Host ""
Write-Host "📝 即将提交的文件列表：" -ForegroundColor Cyan
git status --short

Write-Host ""
$confirmation2 = Read-Host "确认提交这些文件？(y/n)"
if ($confirmation2 -ne 'y') {
    Write-Host "❌ 已取消提交" -ForegroundColor Red
    exit 0
}

Write-Host ""
Write-Host "✍️  步骤 3: 提交代码..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 提交代码
git commit -m "feat: 初始化医疗健康 RAG 问答系统

- 实现混合检索（BM25 + BGE-M3 + Reranker）
- 集成 BERT 查询分类器
- 添加多轮 Prompt Engineering 自验证流水线
- 配置 FastAPI + WebSocket 服务
- 添加 Ragas 评估体系
- 安全配置管理（config.ini 已忽略）"

Write-Host "✅ 代码提交成功！" -ForegroundColor Green

Write-Host ""
Write-Host "🔗 步骤 4: 关联 GitHub 仓库..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# 提示用户输入 GitHub 用户名
$githubUsername = Read-Host "请输入你的 GitHub 用户名"
$repoName = Read-Host "请输入仓库名称（默认: health_qa_system）"

if ([string]::IsNullOrWhiteSpace($repoName)) {
    $repoName = "health_qa_system"
}

$repoUrl = "https://github.com/$githubUsername/$repoName.git"

Write-Host ""
Write-Host "📡 仓库地址: $repoUrl" -ForegroundColor Cyan
Write-Host ""

# 检查是否已存在远程仓库
$existingRemote = git remote -v 2>$null
if ($existingRemote) {
    Write-Host "⚠️  检测到已存在的远程仓库" -ForegroundColor Yellow
    $confirmation3 = Read-Host "是否覆盖现有的远程仓库？(y/n)"
    if ($confirmation3 -eq 'y') {
        git remote remove origin
    } else {
        Write-Host "❌ 已取消" -ForegroundColor Red
        exit 0
    }
}

# 添加远程仓库
git remote add origin $repoUrl

Write-Host ""
Write-Host "🚀 步骤 5: 推送到 GitHub..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# 推送代码
git branch -M main
git push -u origin main

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Green
    Write-Host "  🎉 上传成功！" -ForegroundColor Green
    Write-Host "========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "📍 仓库地址: $repoUrl" -ForegroundColor Cyan
    Write-Host "📖 README: $repoUrl/blob/main/README.md" -ForegroundColor Cyan
    Write-Host "📄 上传指南: $repoUrl/blob/main/GITHUB_UPLOAD_GUIDE.md" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "⚠️  重要提醒：" -ForegroundColor Yellow
    Write-Host "  1. 请勿将 config.ini 文件上传到仓库" -ForegroundColor Yellow
    Write-Host "  2. 其他用户需要复制 config.example.ini 并填写自己的密钥" -ForegroundColor Yellow
    Write-Host "  3. 模型文件需要单独下载（见 README.md）" -ForegroundColor Yellow
    Write-Host ""
} else {
    Write-Host ""
    Write-Host " 推送失败！" -ForegroundColor Red
    Write-Host "请检查：" -ForegroundColor Yellow
    Write-Host "  1. GitHub 用户名和仓库名是否正确" -ForegroundColor Yellow
    Write-Host "  2. 是否有权限访问该仓库" -ForegroundColor Yellow
    Write-Host "  3. 网络连接是否正常" -ForegroundColor Yellow
    exit 1
}
