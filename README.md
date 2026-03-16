# my-master-study
修論研究の内容

# title
推論高速化に向けた静的タスク分割とGPU資源割り当て 

# Objectives 
本研究の目的

単一GPU上で実行されるタスクグラフ処理に対して、Taskflow を基盤とし、効率の良い 
kernel の stream 配置と計算資源割当てを静的に決定する手法を設計・実装することである。 

具体的には、既存のコンパイル時カーネルスケジューリング手法である KeSCo の考え方を参考にしつつ、
Taskflow 上で 同時実行可能な kernel 群を適切に stream に配置する方式を実装し、Taskflow における 
GPU 実行の効率化を図る。 

さらに、本研究では Green Context を用いて、推論難度や重要度に応じた計算資源（SM 数）の指定を行う
手法を Taskflow に統合することを目的とする。Taskflow に Green Context を組み込み、kernel–stream 
配置と計算資源配分を一体として静的に計画する手法は、著者の知る限り既存研究では提案されていない。 

これにより、単に GPU の平均性能やスループットを向上させるだけでなく、重要度の高い処理の遅延を抑
制しつつ GPU 資源を有効活用できる 計算リソース割り当て方法を確立することを本研究の目標とする。 

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

本研究の新規性は、Taskflow を基盤とした GPU タスク実行に対し、(a) KeSCo の考え方をもとにした効
率的な kernel–stream 配置を Taskflow 上で具体的に提案・実装し、さらに (b) Green Context による 
SM 数指定を Taskflow に統合した静的計画手法を提案する点にある。 

KeSCo はコンパイル時解析に基づき、依存関係を踏まえた kernel の stream 配置と同期生成により同時
実行を引き出す静的スケジューリングを示しているが、Taskflow のような汎用タスクグラフ実行基盤
（cudaFlow を含む）に対して、Taskflow 由来の DFG を入力に kernel–stream 配置を体系化して提案す
る枠組みは十分に整備されていない[2][3][6]。 

また Taskflow は優先度を中核機能として提供しておらず、優先度導入が議論されている状況からも、重要
度・遅延制約を明示的に扱う設計が不足している[6]。本研究はこの不足を補うため、Taskflow の DFG を
入力として同時実行単位（PP-set）と stream 分割を決め、重要度に基づく優先度設計と資源配分を静的計
画として統合する[2][3][6][7]。 

さらに、CUDA の実行モデルでは stream 優先度は提供される一方で厳密な遅延保証が難しく[4]、従来は 
stream に対して SM を直接予約する仕組みが一般的に提供されないため、occupancy 制御等による間接制
御が重要となる[5]。これに対し近年導入された Green Context により、コンテキスト単位で利用 SM 数
を指定できるようになった点を踏まえ、本研究は「Taskflow＋Green Context」を統合して 計算難易度（重
要度）に応じた SM 数指定を静的計画に組み込むことを新たに提案する[3][4][5]。 

重要性 

本研究の重要性は、同時カーネル実行の効果（計算型／メモリ型の組合せによる資源相互補完）を活かしつ
つ[1]、動的制御に依存せず、Taskflow のタスクグラフ情報を用いた静的計画として「効率的な kernel
stream 配置」と「重要度に応じた資源配分」を一体で実現する点にある[2][3][6]。 

同時実行は資源競合により重要処理を遅延させ得るため[1]、重要度や遅延要求を踏まえた計画が不可欠で
あるが、既存の静的スケジューリング研究は平均性能やスループット向上を主に扱い、重要タスク保護まで
踏み込めていない課題がある[1][2]。本研究はこのギャップに対し、Taskflow の実行基盤上で重要度を反
映した実行計画を構築できることを示し、深層学習推論やリアルタイム画像処理など低遅延が品質に直結す
る領域での応答安定化と GPU 資源の有効活用（稼働率向上）に貢献する[1][2][3]。 

また、CUDA の仕様上の制約（優先度のみでは遅延保証が難しい、資源配分は間接制御になりやすい）を前
提に[4][5]、Green Context による SM 数指定と occupancy 制御を組み合わせて 実運用上の「資源配分」
を具体化する点は、実システム適用に向けた実装可能性の観点でも重要である[4][5]。 

# Originality/Significance

新規性 

本研究の新規性は、Taskflow を基盤とした GPU タスク実行に対し、(a) KeSCo の考え方をもとにした効
率的な kernel–stream 配置を Taskflow 上で具体的に提案・実装し、さらに (b) Green Context による 
SM 数指定を Taskflow に統合した静的計画手法を提案する点にある。 

KeSCo はコンパイル時解析に基づき、依存関係を踏まえた kernel の stream 配置と同期生成により同時
実行を引き出す静的スケジューリングを示しているが、Taskflow のような汎用タスクグラフ実行基盤
（cudaFlow を含む）に対して、Taskflow 由来の DFG を入力に kernel–stream 配置を体系化して提案す
る枠組みは十分に整備されていない[2][3][6]。 

また Taskflow は優先度を中核機能として提供しておらず、優先度導入が議論されている状況からも、重要
度・遅延制約を明示的に扱う設計が不足している[6]。本研究はこの不足を補うため、Taskflow の DFG を
入力として同時実行単位（PP-set）と stream 分割を決め、重要度に基づく優先度設計と資源配分を静的計
画として統合する[2][3][6][7]。 

さらに、CUDA の実行モデルでは stream 優先度は提供される一方で厳密な遅延保証が難しく[4]、従来は 
stream に対して SM を直接予約する仕組みが一般的に提供されないため、occupancy 制御等による間接制
御が重要となる[5]。これに対し近年導入された Green Context により、コンテキスト単位で利用 SM 数
を指定できるようになった点を踏まえ、本研究は「Taskflow＋Green Context」を統合して 計算難易度（重
要度）に応じた SM 数指定を静的計画に組み込むことを新たに提案する[3][4][5]。 

重要性 

本研究の重要性は、同時カーネル実行の効果（計算型／メモリ型の組合せによる資源相互補完）を活かしつ
つ[1]、動的制御に依存せず、Taskflow のタスクグラフ情報を用いた静的計画として「効率的な kernel
stream 配置」と「重要度に応じた資源配分」を一体で実現する点にある[2][3][6]。 

同時実行は資源競合により重要処理を遅延させ得るため[1]、重要度や遅延要求を踏まえた計画が不可欠で
あるが、既存の静的スケジューリング研究は平均性能やスループット向上を主に扱い、重要タスク保護まで
踏み込めていない課題がある[1][2]。本研究はこのギャップに対し、Taskflow の実行基盤上で重要度を反
映した実行計画を構築できることを示し、深層学習推論やリアルタイム画像処理など低遅延が品質に直結す
る領域での応答安定化と GPU 資源の有効活用（稼働率向上）に貢献する[1][2][3]。 

また、CUDA の仕様上の制約（優先度のみでは遅延保証が難しい、資源配分は間接制御になりやすい）を前
提に[4][5]、Green Context による SM 数指定と occupancy 制御を組み合わせて 実運用上の「資源配分」
を具体化する点は、実システム適用に向けた実装可能性の観点でも重要である[4][5]。 

# Methodology and Evaluation 
・提案手法の詳細 
提案手法は大きく三段階からなる。 
(i) Stream作成 
(ii) Streamの優先度作成 
(iii) 優先度に基づく計算資源割当て 
 
(i) Stream 作成（kernel–stream 配置） 
GPU 上でカーネルを並列実行するため、CUDA stream を生成し、タスク間の依存関係を満たす範囲で同時実
行可能集合（PP-set）を構成する[2]。割当ては、依存を保ちつつ関連するカーネルを同一 stream に寄せ、
同期・競合を抑える。最大 stream 数は max{DFG の最大幅, 定数} として過度な分割を避ける。 
 
(ii) Stream 優先度決定 
各 stream に優先度を付与し、重要処理を含む stream を優先できるようにする。優先度は、stream 内カ
ーネルの重要度（推論難度）を反映する指標（例：タスク属性の総和／重み付き和）で算出する。 
 
(iii) 優先度に基づく計算資源・メモリ割当て 
CUDA では stream へ SM を直接割り当てられず、優先度だけで遅延保証も難しい[4][5]。そこで本研究で
は、Green Context により優先度に応じた利用可能 SM 数を指定して計算資源を確保する[4][5]。必要に応
じて、低優先度カーネルの長時間占有や Green Context 非対応環境に備え、occupancy 制御を補助的に併
用することも検討する。さらに、優先度に基づいてデータ配置（GPU 常駐／CPU 保持）を静的に決め、メモ
リ帯域・キャッシュ競合を緩和する。 
 
以上により、重要処理の遅延悪化を抑えつつ GPU 稼働率を高め、推論難度に応じた静的 GPU 計算リソー
ス割当てによる推論高速化を実現する。 
 
・検証方法 
提案手法の有効性を検証するため，タスクグラフを持つ標準的な GPU ベンチマーク問題を用い，以下の五
手法を同一条件下で実装・比較する。 

• 逐次実行 

• Taskflow による既存実行方式 

• iGniter に基づく GPU 資源割当て方式 

• HEFT に基づく DAG 静的スケジューリング方式 

• 提案手法 

iGniter に基づく方式は，GPU共有時の性能干渉を考慮した資源割当て手法， 
HEFT に基づく方式は，異種計算環境における代表的な DAG 静的スケジューリング手法として位置づける。
これにより，提案手法を，逐次実行や Taskflow による既存実行方式に加え，GPU資源制御および DAG ス
ケジューリングの代表的手法とも比較する。 

評価指標には，全体実行時間，高速化率，GPU稼働率（SM利用率）を用い，重要度を付与したタスクを含む
場合には重要タスクの実行遅延も評価する。提案手法がこれらの指標のうち少なくとも一つで一貫した改善
を示し，他の指標を著しく悪化させないことをもって，有効性を確認する。特に，全体性能の向上と重要タ
スク遅延の抑制の両立を重点的に検証する。 

# Plan
・ヘテロジーニアス環境とホモジーニアス環境のアルゴリズムとの比較
