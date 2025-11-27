#!/bin/bash
# Generate SL mode for all 116 algorithms

cd "$(dirname "$0")/.."

echo "Generating SL mode for all algorithms..."
echo "This creates model-split versions for privacy-preserving training"
echo ""

total=0
success=0

# Read all SS algorithms and generate SL versions
for ss_file in xlearn/_secretflow/generated/ss_*.py; do
    algo_lower=$(basename "$ss_file" | sed 's/ss_//' | sed 's/.py//')
    
    # Extract sklearn import from SS file to get correct module
    sklearn_import=$(grep "from sklearn\." "$ss_file" | grep "import" | head -1)
    
    if [ ! -z "$sklearn_import" ]; then
        # Parse module.Algorithm from import
        module=$(echo "$sklearn_import" | sed 's/.*from sklearn\./sklearn./' | sed 's/ import.*//')
        algo_class=$(echo "$sklearn_import" | sed 's/.* import //')
        
        full_path="${module}.${algo_class}"
        
        ((total++))
        
        if python xlearn/_secretflow/algorithm_migrator_standalone.py \
            --algorithm "$full_path" --mode sl 2>&1 | grep -q "生成适配器"; then
            ((success++))
            echo "[$success/$total] ✅ $algo_lower"
        else
            echo "[$total] ⚠️  $algo_lower (skipped)"
        fi
    fi
done

echo ""
echo "======================================================================"
echo "SL Generation Complete!"
echo "======================================================================"
echo "✅ Success: $success/$total algorithms"
echo ""
echo "📊 Final totals:"
echo "  SS algorithms: $(ls xlearn/_secretflow/generated/ss_*.py 2>/dev/null | wc -l)"
echo "  FL algorithms: $(ls xlearn/_secretflow/generated/fl_*.py 2>/dev/null | wc -l)"
echo "  SL algorithms: $(ls xlearn/_secretflow/generated/sl_*.py 2>/dev/null | wc -l)"
total_algos=$(($(ls xlearn/_secretflow/generated/ss_*.py 2>/dev/null | wc -l) + $(ls xlearn/_secretflow/generated/fl_*.py 2>/dev/null | wc -l) + $(ls xlearn/_secretflow/generated/sl_*.py 2>/dev/null | wc -l)))
echo "  Total: $total_algos implementations"
echo ""

