# my-master-study
修論研究の内容

# title
推論高速化に向けた静的タスク分割とGPU資源割り当て 

# Objectives 
目標 

AI推論高速化 

目的 

Taskflow を基盤として，重要度の高い処理の遅延を抑制しつつ，GPU 資源を有効活用できる静的実行計画
手法を設計・実装 
具体的内容 
・既存のコンパイル時カーネルスケジューリング手法である KeSCo の考え方を参考にしつつ，Taskflow 上
で同時実行可能な kernel 群を適切な stream に配置する方式の提案・実装・評価 
・Green Context を用いて，推論難度や重要度に応じた計算資源（SM 数）の指定を行う手法を Taskflow 
に統合する方式の提案・実装・評価 

# Background 
背景

近年，AI推論は多様な情報処理基盤で利用されており，限られた計算資源のもとで高速かつ安定に実行す
ることが強く求められている。とくに実運用環境では，単一の処理を順に実行するだけでなく，複数の演算
や複数の推論要求を同時に扱う場面が増えており，処理全体をどのように制御するかが性能を大きく左右す
る。こうした処理は，演算間の依存関係をもつ有向非巡回グラフ（DAG: Directed Acyclic Graph）として
表現することで，各演算をノード，データ依存をエッジとして整理でき，並列化可能部分と逐次実行が必要
な部分を明示的に扱える[8][9]。実際，異種計算環境における静的タスクスケジューリング研究では，DAG
に基づいてタスクの優先度付け，割当て，実行順序を決定することが，処理全体の完了時間短縮に有効であ
ることが示されている[9]。  

一方で，AI推論をGPU上で実行する場合，単にDAG上の依存関係を満たすだけでは十分ではない。GPU共有
環境では，複数の処理を同時配置すると，カーネルスケジューリング遅延，L2キャッシュ競合，電力制約に
伴う周波数低下などにより性能干渉が発生し，推論遅延が増大することが報告されている[8]。iGniter で
は，GPU上に複数ワークロードを共存させた際に，こうした干渉を陽に考慮しない資源割当てでは予測可能
な性能保証が難しいことが示されている[8]。このことは，AI推論高速化において，個々の演算を高速化す
るだけでなく，DAG全体の実行順序とGPU資源割当てを一体として設計する必要があることを示している。  
この問題に対し，GPU上で複数カーネルを同時実行し，未利用資源を相互補完的に活用する手法が提案され
ている。たとえば，計算資源を主に消費するカーネルとメモリ帯域を主に消費するカーネルを適切に組み合
わせることで，GPU利用率を高めて実行時間を短縮できることが報告されている[1]。また，KeSCo は，複数
タスクをもつGPUアプリケーションに対し，コンパイル時に依存関係を解析してカーネルの並列実行計画を
構成する静的手法を提案しており，実行時オーバヘッドを抑えながら同時実行性を高められることを示して
いる[4]。しかし，同時実行は常に有効ではなく，組合せや投入順序を誤ると資源競合によって重要処理の
遅延が増大するため，依存関係と処理特性の両方を考慮した実行計画が必要となる[8][1][2]。  

さらに，DAGに基づく静的スケジューリングは，もともと異種計算環境において重要な研究課題として扱わ
れてきた。HEFT や CPOP に代表される手法は，DAG上の上向きランクやクリティカルパスを用いて優先度
を決め，限られたプロセッサ群へタスクを割り当てることで，良好なスケジュール品質と低い計算量を両立
している[9]。この考え方は，AI推論のように依存関係が明確で，同様の処理手順が繰り返し現れるワーク
ロードに対しても有効であり，実行前にGPU上の実行順序や並列化単位を決定する基盤として有望である
[9]。ただし，従来の静的スケジューリング研究の多くは，平均的な完了時間短縮やスループット向上を目
的としており，GPU上での性能干渉や遅延要求の異なる処理の混在を前提とした設計までは十分に扱ってい
ない[8][9]。  

このようなワークロードを扱う実行基盤として，Taskflow は，タスクの依存関係をタスクグラフとして簡
潔に表現でき，CPU/GPUを含む異種環境でも軽量に実行できるという利点を持つ[3]。また，Taskflow の GPU 
向け実行機構である cudaFlow は，CUDA Graph を活用してGPU実行制御を事前に構築・再利用できるため，
ホスト側オーバヘッド削減にも有効である[6]。一方で，Taskflow はタスク優先度を中核機能として十分に
は備えておらず，重要度や遅延制約を明示的に扱う枠組みは限定的である[7]。そのため，Taskflow が表現
する DFG 情報を活用しつつ，GPU上での並列実行単位，stream 分割，優先度設計，資源割当てを統合的に
扱う拡張が必要である。  

加えて，CUDA の実行モデルにも制約がある。CUDA は stream 優先度を提供するが，実行中カーネルを割り
込まないため，優先度設定のみで厳密な遅延保証を行うことは難しい[4]。また従来，stream 単位で SM な
どの計算資源を直接予約する仕組みは一般的ではなく，占有率制御を通じた間接的な調整に依存する必要が
あった[4][5]。近年では Green Context により，コンテキスト単位で利用可能な SM 数を指定する仕組み
が導入され，GPU資源の空間的分割が可能になりつつあるが，依存関係を保ちながらどの処理にどの資源を
与えるかは依然として実行計画に委ねられる[4][5]。 

したがって，低遅延が求められるAI推論においては，単なるカーネル同時実行や局所的な優先度制御では
なく，DAGとして表現された推論処理全体を対象に，依存関係，性能干渉，重要度，GPU資源制約を統合的
に考慮した静的実行計画を構築することが重要な研究課題である。特に，Taskflow が提供する DFG 情報
を活用し，並列実行単位の構成，stream 分割，優先度設計，さらに Green Context や occupancy 制御を
組み合わせた GPU 実行計画を設計することで，AI推論の高速化と重要処理の遅延抑制を両立できる可能性
がある[2][3][8][9]。 

[1] S.-Kazem Shekofteh, Hamid Noori, Mahmoud Naghibzadeh, Holger Fröning, and Hadi Sadoghi Yazdi, 
“cCUDA: Effective Co-Scheduling of Concurrent Kernels on GPUs,” IEEE Transactions on Parallel 
and Distributed Systems, vol. 31, no. 4, Apr. 2020. 

[2] Zejia Lin, Zewei Mo, Xuanteng Huang, Xianwei Zhang, and Yutong Lu, “KeSCo: Compiler-based 
Kernel Scheduling for Multi-task GPU Applications,” in 2023 IEEE 41st International Conference 
on Computer Design (ICCD), 2023. 

[3] Tsung-Wei Huang, Dian-Lun Lin, Chun-Xun Lin, and Yibo Lin, “Taskflow: A Lightweight Parallel 
and Heterogeneous Task Graph Computing System,” IEEE Transactions on Parallel and Distributed 
Systems, vol. 33, no. 6, June 2022. 

[4] NVIDIA Corporation, “CUDA C++ Programming Guide,” CUDA Toolkit Documentation v13.1, Dec. 
2025. 

[5] NVIDIA Corporation, “CUDA Runtime API,” CUDA Toolkit Documentation v13.1.0, Dec. 2025. 

[6] Taskflow Contributors, “A General-purpose Task-parallel Programming System | Taskflow 
QuickStart,” section “Offload Tasks to a GPU,” Taskflow documentation. 

[7] robinchrist, “Allow adding priorities to tasks,” taskflow/taskflow Issue #232, GitHub, 
Sep. 29, 2020. 

[8] Fei Xu, Jianian Xu, Jiabin Chen, Li Chen, Ruitao Shang, Zhi Zhou, and Fangming Liu, “iGniter: 
Interference-Aware GPU Resource Provisioning for Predictable DNN Inference in the Cloud,” IEEE 
Transactions on Parallel and Distributed Systems, vol. 34, no. 3, pp. 812–825, Mar. 2023. 

[9] Haluk Topcuoglu, Salim Hariri, and Min-You Wu, “Performance-Effective and Low-Complexity 
Task Scheduling for Heterogeneous Computing,” IEEE Transactions on Parallel and Distributed 
Systems, vol. 13, no. 3, pp. 260–274, Mar. 2002. 

# Originality/Significance 

新規性 

本研究の新規性は，Taskflow を基盤とした GPU タスク実行に対し， 
(1) KeSCo の考え方を参考に，Taskflow 由来の DFG を入力として効率的な kernel–stream 配置を静的に
設計する手法を提案 
(2) Green Context による SM 数指定を Taskflow に統合し，重要度に応じた資源配分を含む静的実行計
画を提案 
 
KeSCo はコンパイル時解析に基づく kernel 配置を示しているが，Taskflow のような汎用タスクグラフ実
行基盤上で，DFG を用いた kernel–stream 配置を体系的に扱う枠組みは十分に整備されていない
[2][3][6]。 
また，Taskflow は優先度を中核機能として十分に備えておらず，重要度や遅延制約を明示的に扱う設計が
不足している[6][7]。さらに，Green Context を用いて計算資源配分まで含めた静的計画を Taskflow 上で
実現する提案は，著者の知る限り既存研究には見当たらない[3][4][5]。 
 
重要性 

本研究の重要性は，Taskflow のタスクグラフ情報を用いた静的計画として，効率的な kernel–stream 配置
と重要度に応じた GPU 資源配分を一体的に実現しようとする点にある[2][3][6]。 
同時カーネル実行は，計算型処理とメモリ型処理の組合せによって GPU 利用率向上に有効である一方[1]，
資源競合により重要処理の遅延を悪化させる可能性がある[1][8]。このため，平均性能やスループットの向
上だけでなく，重要処理の遅延抑制まで考慮した実行計画が必要である。 
本研究はこの課題に対し，Taskflow 上で重要度を反映した静的実行計画を実現することで，深層学習推論
やリアルタイム画像処理など，低遅延性と安定性が要求される実応用に貢献する。とくに，映像解析，自動
運転支援，ロボット制御，対話AI などでは，重要な推論要求に対する応答遅延の抑制が品質や安全性に直
結するため，本研究はサービス品質の向上と GPU 資源の有効活用の両面で意義をもつ。さらに，単一GPU
環境を対象とすることで，エッジ機器や単一アクセラレータ搭載サーバなど実際の利用環境へ適用しやす
く，実装可能性の高い資源制御手法を与える点でも重要である。加えて，CUDA の既存機構だけでは厳密な
遅延保証や資源配分が難しいという制約のもとで[4][5]，Green Context による SM 数指定を活用し，実装
可能な形で資源制御を具体化する点にも意義がある[4][5]。  


# Methodology and Evaluation 
・提案手法の詳細 
提案手法は以下の4段階からなる。 
(i)     Construct a DFG 
(ii)     DFG-based task leveling 
(iii) Kernel-to-stream assignment 
(iv)     Stream resource allocation 
 
(i)    Construct a DFG 
目的：タスク間の依存関係を明確にする。 
各タスクの先行・後続関係を整理し、処理全体を Data Flow Graph（DFG） として表現する。 
DFG は、各タスクの依存関係と実行順序を表すグラフである。 
 
(ii)   DFG-based task leveling 
目的：実行順序を整理し、並列実行可能なタスク群を明らかにする。 
DFGに基づいてタスクをレベル分けし、依存関係を保ちながら実行可能な単位に整理する。 
 
(iii)  Kernel-to-stream assignment 
目的：各カーネルを適切なstreamに割り当て、並列実行性を高める。 
タスクの依存関係やレベル構造を踏まえ、各kernelをどのstreamで実行するか決定する。 
 
(iv)   Stream resource allocation 
目的：各streamにGPU計算資源を配分し、処理効率を高める。 
各streamの重要度や負荷に応じて計算資源を割り当て、全体の性能向上を図る。 
 
以上により、重要処理の遅延悪化を抑えつつGPU 稼働率を高め、 
推論難度に応じた静的 GPU計算リソース割当てによる推論高速化を実現する。 
 
・検証方法 
提案手法の有効性を検証するため、タスクグラフを持つ標準的な GPU ベンチマーク問題を用い、以下の三
手法を同一条件下で実装・比較する。 

• 逐次実行 

• Taskflow による既存実行方式 

• iGniter に基づく GPU 資源割当て方式 

• HEFT に基づく DAG 静的スケジューリング方式 

• 提案手法 
 
iGniter に基づく方式は，GPU共有時の性能干渉を考慮した資源割当て手法として位置づける[8]。 
また，HEFT に基づく方式は，異種計算環境における代表的な DAG 静的スケジューリング手法として位置
づける[9]。これにより，提案手法を，逐次実行や Taskflow による既存実行方式[3][6]だけでなく，GPU 資
源制御および DAG スケジューリングに関する代表的既存手法とも比較できる。 

評価指標としては，全体実行時間，高速化率，GPU 稼働率（SM 利用率）を用いる。 

加えて，重要度を付与したタスクを含むベンチマークでは，重要タスクの実行遅延も評価する。提案手法が
少なくとも一つの指標で一貫した改善を示し，かつ他の指標を著しく悪化させないことをもって，有効性を
確認する。とくに，全体性能の向上と重要タスク遅延の抑制の両立を重点的に検証する。・ヘテロジーニアス環境とホモジーニアス環境のアルゴリズムとの比較

・実装環境 

GPU：NVIDIA H100 PCIe 

GPUメモリ：81559 MiB（約80 GiB） 

CPU：AMD EPYC 7313 16-Core Processor ×2（計32コア） 

主記憶：125 GiB 

OS：Linux 

開発環境：CUDA Toolkit 13.1，NVIDIA Driver，CMake 

計測・プロファイリング： 

・Nsight Systems（nsys） 

・Nsight Compute（ncu） 

・CUDA Events（実行時間計測） 

使用言語：C++（C++17/20），CUDA 

タスクグラフ実行基盤：Taskflow（cudaFlow を含む） 

GPU実行基盤：CUDA Runtime / CUDA Toolkit 

GPU資源制御機構：Green Context 
