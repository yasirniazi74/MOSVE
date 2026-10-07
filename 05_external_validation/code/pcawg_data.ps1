param(
    [ValidateSet("Inventory", "CoreData")]
    [string]$Action = "Inventory"
)

$ErrorActionPreference = "Stop"
$ValidationDir = $PSScriptRoot | Split-Path -Parent
$ManifestDir = Join-Path $ValidationDir "01_manifests"
$RawDir = Join-Path $ValidationDir "02_raw_data"
$AwsExe = (Get-Command aws -ErrorAction Stop).Source
$Endpoint = "https://object.genomeinformatics.org"

New-Item -ItemType Directory -Path $ManifestDir, $RawDir -Force | Out-Null

function Invoke-OpenS3 {
    param([string[]]$Arguments, [string]$FailureMessage)
    & $AwsExe @Arguments
    if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
}

function Save-S3Listing {
    param([string]$Uri, [string]$OutputName, [switch]$Recursive)
    $arguments = @("s3", "ls", $Uri)
    if ($Recursive) { $arguments += "--recursive" }
    $arguments += @("--endpoint-url", $Endpoint, "--no-sign-request")
    Invoke-OpenS3 $arguments "Could not list $Uri" |
        Set-Content -LiteralPath (Join-Path $ManifestDir $OutputName)
}

function Download-OpenFile {
    param([string]$Uri, [string]$Destination)
    $parent = Split-Path -Parent $Destination
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    if (Test-Path -LiteralPath $Destination) {
        Write-Output "Already present: $Destination"
        return
    }
    $arguments = @("s3", "cp", $Uri, $Destination, "--endpoint-url", $Endpoint, "--no-sign-request")
    Invoke-OpenS3 $arguments "Could not download $Uri" | Out-Null
}

if ($Action -eq "Inventory") {
    foreach ($directory in @(
        (Join-Path $RawDir "metadata"),
        (Join-Path $RawDir "driver_labels"),
        (Join-Path $ValidationDir "03_processed_data"),
        (Join-Path $ValidationDir "04_features"),
        (Join-Path $ValidationDir "05_labels"),
        (Join-Path $ValidationDir "06_models"),
        (Join-Path $ValidationDir "07_results")
    )) {
        New-Item -ItemType Directory -Path $directory -Force | Out-Null
    }

    Save-S3Listing "s3://icgc25k-open/PCAWG/" "pcawg_top_level.txt"
    Save-S3Listing "s3://icgc25k-open/PCAWG/consensus_snv_indel/" "mutation_files.txt"
    Save-S3Listing "s3://icgc25k-open/PCAWG/consensus_cnv/" "cnv_files.txt" -Recursive
    Save-S3Listing "s3://icgc25k-open/PCAWG/transcriptome/gene_expression/" "expression_files.txt"
    Save-S3Listing "s3://icgc25k-open/PCAWG/driver_mutations/" "driver_label_files.txt"
    Save-S3Listing "s3://icgc25k-open/PCAWG/donors_and_biospecimens/" "sample_metadata_files.txt"

    Download-OpenFile "s3://icgc25k-open/PCAWG/donors_and_biospecimens/pcawg_sample_sheet.tsv" (Join-Path $RawDir "metadata\pcawg_sample_sheet.tsv")
    Download-OpenFile "s3://icgc25k-open/PCAWG/donors_and_biospecimens/pcawg-wgs-rnaseq-mirna.xlsx" (Join-Path $RawDir "metadata\pcawg-wgs-rnaseq-mirna.xlsx")
    Download-OpenFile "s3://icgc25k-open/PCAWG/driver_mutations/TableS1_compendium_mutational_drivers.xlsx" (Join-Path $RawDir "driver_labels\TableS1_compendium_mutational_drivers.xlsx")
    Download-OpenFile "s3://icgc25k-open/PCAWG/driver_mutations/TableS3_panorama_driver_mutations_ICGC_samples.public.tsv.gz" (Join-Path $RawDir "driver_labels\TableS3_panorama_driver_mutations_ICGC_samples.public.tsv.gz")

    [ordered]@{
        inventory_created = (Get-Date).ToString("o")
        source = "ICGC 25K open object bucket / PCAWG"
        endpoint = $Endpoint
        large_files_downloaded = $false
        mutation_candidate = "PCAWG/consensus_snv_indel/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz"
        cnv_candidate = "PCAWG/consensus_cnv/gene_level_calls/all_samples.consensus_CN.by_gene.170214.txt.gz"
        expression_candidate = "PCAWG/transcriptome/gene_expression/tophat_star_fpkm_uq.v2_aliquot_gl.tsv.gz"
        methylation_candidate = $null
        note = "PCAWG root has no directly listed harmonized methylation matrix; audit separately."
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $ManifestDir "inventory_summary.json")
    Write-Output "PCAWG inventory complete: $ValidationDir"
}
else {
    Download-OpenFile "s3://icgc25k-open/PCAWG/consensus_snv_indel/final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz" (Join-Path $RawDir "mutation\final_consensus_passonly.snv_mnv_indel.icgc.public.maf.gz")
    Download-OpenFile "s3://icgc25k-open/PCAWG/consensus_cnv/gene_level_calls/all_samples.consensus_CN.by_gene.170214.txt.gz" (Join-Path $RawDir "cnv\all_samples.consensus_CN.by_gene.170214.txt.gz")
    Download-OpenFile "s3://icgc25k-open/PCAWG/consensus_cnv/gene_level_calls/all_samples.consensus_level_calls.by_gene.170214.txt.gz" (Join-Path $RawDir "cnv\all_samples.consensus_level_calls.by_gene.170214.txt.gz")
    Download-OpenFile "s3://icgc25k-open/PCAWG/transcriptome/gene_expression/tophat_star_fpkm_uq.v2_aliquot_gl.tsv.gz" (Join-Path $RawDir "expression\tophat_star_fpkm_uq.v2_aliquot_gl.tsv.gz")

    Get-ChildItem -LiteralPath $RawDir -Recurse -File |
        Select-Object FullName, Length, LastWriteTime |
        Export-Csv -LiteralPath (Join-Path $ManifestDir "downloaded_files.csv") -NoTypeInformation
    Write-Output "PCAWG core downloads complete."
}
