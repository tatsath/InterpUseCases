#!/bin/bash

echo "🚀 Auto-Interp Runner"
echo "====================="
echo ""
echo "Choose your approach:"
echo "1. CLI Version (Small Dataset - 50 latents, 1M tokens)"
echo "2. Programmatic Version (Full Control - FAISS, Finance Labels)"
echo "3. Exit"
echo ""

read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        echo ""
        echo "🚀 Running CLI Version (Small Dataset)..."
        echo "=========================================="
        python run_cli_small.py
        ;;
    2)
        echo ""
        echo "🚀 Running Programmatic Version (Full Control)..."
        echo "================================================"
        python finance_autointerp.py
        ;;
    3)
        echo "👋 Goodbye!"
        exit 0
        ;;
    *)
        echo "❌ Invalid choice. Please run the script again."
        exit 1
        ;;
esac
